import argparse
import json
import torch
import pprint
from tqdm import tqdm
import os
import torch.distributed as dist
import re
from itertools import chain
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
import torch.multiprocessing as mp
from collections import deque
from transformers import AutoModelForCausalLM, AutoTokenizer, get_scheduler
import queue as queue_module
from queue import Empty


from utils import get_model_tokenizer, get_loaded_model_tokenizer, setup, cleanup, log_args, gather_and_merge_dicts, gather_and_merge_list
from utils_adastar import get_dataloader_adastar, update_minheap_winlose_front, log_stable_minheap_new

pp = pprint.PrettyPrinter(indent=2).pprint

def write_new_data(args, pred, data, endoftext):
    if args.task == "arc_challenge":
        q, choices = data["question"], data["choices"]
        options_text = "\n".join([f"{label}. {opt}" for label, opt in zip(choices["label"], choices["text"])])
        new_example = f"Q: {q}\nOptions:\n{options_text}\nA: {pred}" + endoftext
    elif args.task == "cqa":
        text = data["question"]
        q = text['stem']
        choices = text['choices']
        options_text = "\n".join([f'({choice["label"]}) {choice["text"]}' for choice in choices])
        new_example = f"Q: {q}\nOptions:\n{options_text}\nA: {pred}" + endoftext
    elif args.task == "gsm8k":
        q = data["question"]
        new_example = f"Q: {q}\nA: {pred}" + endoftext
    elif args.task == "svamp":
        q = data["question_concat"]
        new_example = f"Q: {q}\nA: {pred}" + endoftext
    elif args.task == "mate":
        q = data["question_concat"]
        new_example = f"Q: {q}\nA: {pred}" + endoftext
    elif args.task == "cladder":
        q = data["question"]
        new_example = f"Q: {q}\nA: {pred}" + endoftext
    elif args.task == "anli_r1":
        t1 = data["premise"]
        choices = data['choices']
        labels = choices["label"]
        texts = choices["text"]
        options_text = "\n".join([f'({label}) {text}' for label, text in zip(labels, texts)])

        t2 = data["hypothesis"]
        q = f"{t1} {t2}"
        new_example = f"Q: {q}\nOptions:\n{options_text}\nA: {pred}" + endoftext
    else:
        raise NotImplementedError

    return new_example

def test_metric_STaR(args, predictions, datas, tokenizer, rank):
    '''
    채점하고 맞게 푼 내용, 맞춘 문제번호, 배치단위 정답결과 반환
    '''
    # 정답 데이터 저장용
    correct_str = []
    # 맞춘 문제번호 저장용
    correct_data_idx = []

    # 맞춘문제 플래그
    correct = [False for _ in range(len(predictions))]

    for idx, (pred, data) in enumerate(zip(predictions, datas), 1):
        cur_correct = False
        answer = data["answer"]

        # 답변하고도 Q: 이런식으로 지가 질문을 이어만드는 repetition이 있는 경우가 있어서 , repetition 영향을 제거함
        q_start_idx = pred.find("Q: ")
        if q_start_idx != -1:
            pred = pred[:q_start_idx]

        # task중에 "… reasoning … #### 5\nblah blah" 이런식으로 ####포멧 뒤에 답을 출력하는애들이 있어서 그 경우 처리해줌
        if "####" in pred:
            parts = pred.split("####")
            # #### 뒤에 5\nblah blah 이런식으로 있으면 규칙기반으로 파싱. 근데 이 방식은 좀 불안정할듯
            if len(parts) > 1 and len(parts[1].split()) > 0:
                pred = parts[0] + "#### " + parts[1].split()[0]
            # #### 뒤에 뭐 없으면 원상복구
            else:
                pred = parts[0] + "#### "

        if args.task == "arc_challenge":
            matches = list(re.finditer(r"\b(A|B|C|D)\b", pred))
            pred_answer = matches[-1].group(1) if matches else None

        elif args.task == "cladder":
            matches = re.findall(r"\b(yes|no)\b", pred, re.IGNORECASE)
            pred_answer = matches[-1].lower() if matches else None

        elif args.task == "cqa":
            matches = list(re.finditer(r"\b(A|B|C|D|E)\b", pred))
            pred_answer = matches[-1].group(1) if matches else None
        
        elif args.task == "svamp":
            # -34.21 이런거 잡아내게 정규표현식 설정
            matches = re.findall(r"-?\d+\.?\d*", pred)
            # 마지막에 나온 숫자를 정답일것이라고 가정하고 추출
            pred_answer = matches[-1] if matches else None
            # 방어적으로 answer도 정규표현식으로 재추출
            ref_match = re.search(r"-?\d+\.?\d*", str(answer))
            ref_answer = ref_match.group(0) if ref_match else None
            if pred_answer == ref_answer:
                cur_correct = True

        elif args.task == "mate":
            matches = re.findall(r"-?\d+\.?\d*", pred)
            pred_answer = matches[-1] if matches else None
            ref_match = re.search(r"-?\d+\.?\d*", str(answer))
            ref_answer = ref_match.group(0) if ref_match else None
            if pred_answer == ref_answer:
                cur_correct = True

        elif args.task == "anli_r1":
            matches = list(re.finditer(r"\b(0|1|2)\b", pred))
            pred_answer = matches[-1].group(1) if matches else None

        else:
            matches = re.findall(r"-?\d+\.?\d*", pred)
            pred_answer = matches[-1] if matches else None
            ref_match = re.search(r"####\s*([-+]?\d*\.?\d+)", str(answer))
            ref_answer = ref_match.group(1).strip() if ref_match else None

            if pred_answer == ref_answer:
                cur_correct = True
                
        if args.task == "arc_challenge" and pred_answer and pred_answer == answer:
            cur_correct = True
        elif args.task == "cqa" and pred_answer and pred_answer == answer:
            cur_correct = True
        elif args.task == "cladder" and pred_answer and pred_answer == answer.lower():
            cur_correct = True
        elif args.task == "anli_r1" and pred_answer and str(pred_answer) == str(answer):
            cur_correct = True
        
        # cur_correct가 true이면, 3개의 리스트 업데이트
        if cur_correct:
            correct[idx-1] = True
            correct_str.append(write_new_data(args, pred, data, tokenizer.eos_token))
            correct_data_idx.append(data["idx"])

    return correct_str, correct_data_idx, correct

def broadcast_list(data, src_rank):
    object_list = [data if dist.get_rank() == src_rank else None]
    dist.broadcast_object_list(object_list, src=src_rank)
    return object_list[0]

# task별로 Q:, options: 같은 문제포멧을 맞춰주고, n-shot 프롬프트 붙여준 후에 토큰화함. 힌트주는 로직도 여기 있는듯
def prompt_preprocess(args, examples, tokenizer, n_shot_prompts, n_shot_prompts_hint, hint):
    combined_texts = []

    if args.task == "arc_challenge":
        for i in range(len(examples)):
            q, choices, a = examples[i]["question"], examples[i]["choices"], examples[i]["answerKey"]
            options_text = "\n".join([f"({label}). {opt}" for label, opt in zip(choices["label"], choices["text"])])
            if hint[i] and args.no_hint == False:
                combined_texts.append(f"{n_shot_prompts_hint}\nQ: {q} ({a})\nOptions:\n{options_text}\nA: ")
            else:
                combined_texts.append(f"{n_shot_prompts}\nQ: {q}\nOptions:\n{options_text}\nA: ")

    elif args.task == "cqa":
        for i in range(len(examples)):
            q = examples[i]["question"]['stem']
            choices = examples[i]["question"]['choices']
            ans = examples[i]["answerKey"]
            options_text = "\n".join([f'({choice["label"]}) {choice["text"]}' for choice in choices])

            if hint[i] and args.no_hint == False:
                combined_texts.append(f"{n_shot_prompts_hint}\nQ: {q} ({ans})\nOptions:\n{options_text}\nA: ")
            else:
                combined_texts.append(f"{n_shot_prompts}\nQ: {q}\nOptions:\n{options_text}\nA: ")

    elif args.task == "gsm8k":
        for i in range(len(examples)):
            q = examples[i]["question"]
            a = examples[i]["answer"]
            answer = a.split()[-1]
            if hint[i] and args.no_hint == False:
                combined_texts.append(f"{n_shot_prompts_hint}\nQ: {q} ({answer})\nA: ")
            else:
                combined_texts.append(f"{n_shot_prompts}\nQ: {q}\nA: ")

    elif args.task == "svamp":
        for i in range(len(examples)):
            q = examples[i]["question_concat"]
            a = examples[i]["answer"]
            if hint[i] and args.no_hint == False:
                combined_texts.append(f"{n_shot_prompts_hint}\nQ: {q} ({a})\nA: ")
            else:
                combined_texts.append(f"{n_shot_prompts}\nQ: {q}\nA: ")

    elif args.task == "mate":
        for i in range(len(examples)):
            q = examples[i]["question_concat"]
            a = examples[i]["answer"]
            if hint[i] and args.no_hint == False:
                combined_texts.append(f"{n_shot_prompts_hint}\nQ: {q} ({a})\nA: ")
            else:
                combined_texts.append(f"{n_shot_prompts}\nQ: {q}\nA: ")

    elif args.task == "cladder":
        for i in range(len(examples)):
            q = examples[i]["question"]
            a = examples[i]["answer"]
            if hint[i] and args.no_hint == False:
                combined_texts.append(f"{n_shot_prompts_hint}\nQ: {q} ({a})\nA: ")
            else:
                combined_texts.append(f"{n_shot_prompts}\nQ: {q}\nA: ")

    elif args.task == "anli_r1":
        for i in range(len(examples)):
            q = examples[i]["premise"]
            h = examples[i]["hypothesis"]
            choices = examples[i]["choices"]
            a = examples[i]["label"]  
            labels = choices["label"]
            texts = choices["text"]
            options_text = "\n".join([f'({label}) {text}' for label, text in zip(labels, texts)])
            if hint[i] and args.no_hint == False:
                combined_texts.append(f"{n_shot_prompts_hint}\nQ: {q} {h} ({a})\nOptions:\n{options_text}\nA: ")
            else:
                combined_texts.append(f"{n_shot_prompts}\nQ: {q} {h}\nOptions:\n{options_text}\nA: ")

    else:
        raise NotImplementedError

    tokenized = tokenizer(
        combined_texts,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=args.max_length,
    )
    
    return tokenized

def eval_examples(args, model, rank, data_queue, tokenizer, gen_length, n_shot_prompts, n_shot_prompts_hint):
    generate_fn = model.module.generate if hasattr(model, 'module') else model.generate
    world_size = dist.get_world_size()

    correct_total = torch.zeros(4).to(rank)
    gpu_correct_total = torch.zeros(4).to(rank)

    pbar = tqdm(total=args.required_data_num, desc="Processing", disable=(rank != 0))

    gpu_rationale_dataset = []
    gpu_data_idx = []
    gpu_winlose = {}
    buffer = []
    hint = []
    data_depleted = torch.zeros(1).to(rank)

    def fill_buffer():
        while len(buffer) < args.batch_size:
            try:
                new_data = data_queue.get(timeout=2)
                if new_data is None:
                    break
                buffer.append(new_data)
                hint.append(False)
                id_idx = new_data["idx"]
                gpu_winlose[f'id_{id_idx}'] = {"iter": args.exp_iter, "win": 0, "total": 0}
            except (queue_module.Empty, EOFError, OSError) as e:
                break

    fill_buffer()

    batch_idx = 0
    
    with torch.no_grad():
        while True:
            fill_buffer()

            if len(buffer) == 0:
                data_depleted[0] = 1
            else:
                tokenized = prompt_preprocess(args, buffer, tokenizer, n_shot_prompts, n_shot_prompts_hint, hint)
                input_ids = tokenized["input_ids"].to(rank)
                attention_mask = tokenized["attention_mask"].to(rank)

                outputs = generate_fn(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=gen_length,
                    pad_token_id=tokenizer.eos_token_id,
                    do_sample=True,
                    top_p = 0.9,
                    temperature=args.inference_temp
                )

                generated_tokens = outputs[:, input_ids.shape[-1]:]
                predictions = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)

                eos_token_id = tokenizer.eos_token_id
                actual_lengths = 0
                actual_c=0
                for seq in generated_tokens:
                    eos_positions = (seq == eos_token_id).nonzero(as_tuple=True)[0]
                    if eos_positions.numel() > 0:
                        actual_length = eos_positions[0].item() + 1
                    else:
                        actual_length = seq.size(0)
                    actual_lengths+=actual_length
                    actual_c+=1

                actual_gen_len = actual_lengths
                
                flops_log_file = f"{args.idx_save}/flops_log_{rank}.json"
                log_args(flops_log_file, iter=args.exp_iter,idx=batch_idx, split="inf", hint=hint , batch=actual_c, input= actual_c*input_ids.size(1),output= actual_gen_len)
                batch_idx += 1

                correct_str, correct_data_idx, correct = test_metric_STaR(args, predictions, buffer, tokenizer, rank)
                
                for i in range(len(buffer)):
                    if hint[i]:
                        gpu_correct_total[3] += 1
                        if correct[i]:
                            gpu_correct_total[1] += 1
                    else:
                        gpu_correct_total[2] += 1
                        if correct[i]:
                            gpu_correct_total[0] += 1

                correct_total = gpu_correct_total.clone()
                gpu_rationale_dataset.extend(correct_str)
                gpu_data_idx.extend(correct_data_idx)

                new_buffer = []
                new_hint = []
                for buffer_idx in range(len(buffer)):
                    id_idx = buffer[buffer_idx]["idx"]
                    if correct[buffer_idx]:
                        gpu_winlose[f"id_{id_idx}"]["win"] += 1
                        gpu_winlose[f"id_{id_idx}"]["total"] += 1
                    else:
                        gpu_winlose[f"id_{id_idx}"]["total"] += 1                        
                        if not hint[buffer_idx]:
                            new_buffer.append(buffer[buffer_idx])
                            new_hint.append(True)
                buffer = new_buffer
                hint = new_hint

            dist.barrier()
            dist.all_reduce(correct_total, op=dist.ReduceOp.SUM)
            dist.all_reduce(data_depleted, op=dist.ReduceOp.SUM)

            corrects = int(correct_total[0].item() + correct_total[1].item())
            if rank == 0:
                pbar.n = corrects
                pbar.refresh()

            if corrects >= args.required_data_num:
                break
            if data_depleted[0] == world_size:
                break

    if rank == 0:
        pbar.close()
    dist.barrier()
    return gpu_rationale_dataset, gpu_data_idx, gpu_winlose, correct_total

# 공유 큐에서 문제를 뽑아 GPU 디바이스별로 풀어봄. 논문에선 문제당 몇트할지 K로 조정가능한것같았는데 구현에선 사실상 한문제당 2번으로 고정인듯
def eval_examples_nohint(args, model, rank, data_queue, tokenizer, gen_length, n_shot_prompts, n_shot_prompts_hint):
    '''
    공유 큐에서 문제를 뽑아 GPU 디바이스별로 풀어봄.논문에선 문제당 몇트할지 K로 조정가능한것같았는데 구현에선 사실상 한문제당 2번으로 고정인듯
    '''
    # 생성할때 기존방식대로 forward하면 한 토큰 나와서 LM에서는 generate()로 답변 쫙 생성함
    # 근데 DDP로 감싸면 실제 모델은 model.module로 들어가버려서 generate함수 접근법이 달라짐
    generate_fn = model.module.generate if hasattr(model, 'module') else model.generate
    world_size = dist.get_world_size()

    # 모든 gpu 합산한 결과 담을 텐서. [no-hint에서 맞춘개수, no-hint 전체 시도수, hint에서 맞춘개수, hint 전체 시도수]라서 4칸 
    correct_total = torch.zeros(4).to(rank)
    # 이 gpu에서의 결과 담을 텐서
    gpu_correct_total = torch.zeros(4).to(rank)

    # tqdm 객체 생성
    pbar = tqdm(total=args.required_data_num, desc="Processing", disable=(rank != 0))

    # 정답 데이터셋 잠시 모아두는 리스트
    gpu_rationale_dataset = []
    # 정답 데이터셋 인덱스 기록하는 리스트
    gpu_data_idx = []
    # \tilde{t}, w 통계용 딕셔너리
    gpu_winlose = {}
    # 뽑은 문제들 배치로 만들어서 돌릴려고 임시로 담아놓는 리스트
    buffer = []
    # 2트인지 표시하는 플래그 리스트. 원래 논문에서는 한문제당 K번 풀어보게 되어있는데, 구현에선 사실상 2번 시도로 고정한듯.
    second = []
    # 힙이(멀티프로세싱때문에 구현상으로는 큐) 비었는지 표시하는 플래그 텐서. 비면 1로 바꾸는듯
    # dist 같은 분산 연산은 텐서를 대상으로 하기때문에 플래그를 텐서로 만들었다고 함
    data_depleted = torch.zeros(1).to(rank)

    # 버퍼에 배치크기만큼 추론돌릴 문제들 채워넣는 함수
    def fill_buffer():
        while len(buffer) < args.batch_size:
            try:
                # 큐에서 문제를 뽑는데, 무한대기하면 다른 GPU들도 barrier에서 같이대기하기때문에 2초 시간제한
                new_data = data_queue.get(timeout=2)
                if new_data is None:
                    break
                # 버퍼채우고 second도 채움
                buffer.append(new_data)
                second.append(False)
                id_idx = new_data["idx"]
                # 딕셔너리에 \tilde{t}, w 모두 기록
                gpu_winlose[f'id_{id_idx}'] = {"iter": args.exp_iter, "win": 0, "total": 0}
            except (queue_module.Empty, EOFError, OSError) as e:
                break

    # 아래와 중복된코드라 필요없는듯
    fill_buffer()

    batch_idx = 0

    # 추론모드로 설정하고
    with torch.no_grad():
        while True:
            fill_buffer()

            if len(buffer) == 0:
                data_depleted[0] = 1
            else:
                queue_empty = fill_buffer()
                # 한줄로 문제 만들어주고 토큰화까지 함
                tokenized = prompt_preprocess(args, buffer, tokenizer, n_shot_prompts, n_shot_prompts_hint, second) # (B, 최대길이)
                input_ids = tokenized["input_ids"].to(rank)
                attention_mask = tokenized["attention_mask"].to(rank)

                # 배치단위로 추론 실시. 로짓이 아니라 토큰 id가 나옴
                outputs = generate_fn(  # (B, 프롬프트길이 + 생성된길이)
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=gen_length,
                    pad_token_id=tokenizer.eos_token_id,
                    do_sample=True,
                    top_p = 0.9,
                    temperature=args.inference_temp
                )

                # 생성된 출력에서 알맹이만 뽑아내고 디코딩
                generated_tokens = outputs[:, input_ids.shape[-1]:]
                predictions = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)

                eos_token_id = tokenizer.eos_token_id
                actual_lengths = 0
                actual_c=0
                # 각 생성들을 순회하면서 통계용 수치 기록
                for seq in generated_tokens:
                    # eos인지 불리언텐서 만들고
                    eos_positions = (seq == eos_token_id).nonzero(as_tuple=True)[0]
                    # eos 토큰 여부에 따라 생성된 출력 길이 측정
                    if eos_positions.numel() > 0:
                        actual_length = eos_positions[0].item() + 1
                    else:
                        actual_length = seq.size(0)
                    # 배치단위로 생성된 총 출력길이 누적
                    actual_lengths+=actual_length
                    # 처리한 샘플 수 누적
                    actual_c+=1

                actual_gen_len = actual_lengths
                
                # svamp/svamp_qwen_adastar_new_square_10/svamp_qwen_adastar_new_square_10_19/flops_log_1.json 이런식의 로그파일에 로그한줄 추가
                flops_log_file = f"{args.idx_save}/flops_log_{rank}.json"
                log_args(flops_log_file, iter=args.exp_iter,idx=batch_idx, split="inf", hint=second , batch=actual_c, input= actual_c*input_ids.size(1),output= actual_gen_len)
                batch_idx += 1

                correct_str, correct_data_idx, correct = test_metric_STaR(args, predictions, buffer, tokenizer, rank)
                
                # gpu_correct_total에 로그 업데이트. 2트한다고 총 시도한 횟수를 늘려버리진 않음.
                for i in range(len(buffer)):
                    if second[i]:
                        if correct[i]:
                            gpu_correct_total[0] += 1
                    else:
                        gpu_correct_total[1] += 1
                        if correct[i]:
                            gpu_correct_total[0] += 1

                # gpu_correct_total은 모든 rank 합친버전 만들기 위해 복제해서 독립적인 텐서 만듦
                correct_total = gpu_correct_total.clone()
                gpu_rationale_dataset.extend(correct_str)
                gpu_data_idx.extend(correct_data_idx)

                # \tilde{t}, w 통계용 딕셔너리인 winlose도 업데이트
                new_buffer = []
                new_second = []
                for buffer_idx in range(len(buffer)):
                    id_idx = buffer[buffer_idx]["idx"]
                    # 맞췄으면 winlose 업데이트하고 끝이고
                    if correct[buffer_idx]:
                        gpu_winlose[f"id_{id_idx}"]["win"] += 1
                        gpu_winlose[f"id_{id_idx}"]["total"] += 1
                    # 틀렸는데 1트면 winlose 업데이트한뒤에 다음에 풀 버퍼에 틀린문제 한번 더 담아줌
                    else:
                        gpu_winlose[f"id_{id_idx}"]["total"] += 1                        
                        if not second[buffer_idx]:
                            new_buffer.append(buffer[buffer_idx])
                            new_second.append(True)
                # 다음추론때 쓸 버퍼랑 2트 플래그리스트 설정
                buffer = new_buffer
                second = new_second

            # 다른 GPU도 이 라인에 도착할때까지 기다려서 배치 돌린 수 싱크가 같게 맞춰짐
            dist.barrier()
            # 로깅용 데이터 합치기위해서 분산 연산함
            dist.all_reduce(correct_total, op=dist.ReduceOp.SUM)
            # 이건 몇개의 GPU가 데이터 고갈되었는지 확인.
            dist.all_reduce(data_depleted, op=dist.ReduceOp.SUM)

            corrects = int(correct_total[0].item())
            if rank == 0:
                # pbar에서 몇개 채웠는지로 더해서 업데이트함
                pbar.n = corrects
                pbar.refresh()

            # 맞춰서 채운 데이터 수가 필요한 데이터양보다 많으면 멈춤
            if corrects >= args.required_data_num:
                break
            # 모든 GPU에서 데이터가 고갈돼서 4가 나와야 멈춤
            if data_depleted[0] == world_size:
                break

    if rank == 0:
        pbar.close()
    dist.barrier()
    return gpu_rationale_dataset, gpu_data_idx, gpu_winlose, correct_total

def evaluate(args, model, rank, world_size, data_queue, tokenizer, gen_length, target_save, n_shot_prompts, n_shot_prompts_hint):
    # 모델을 평가모드로 전환하고
    model.eval()

    # split이 train일때만
    if args.split == "train":
        # rank별로 추론돌려서 정답데이터셋 만들고
        if args.no_hint:
            gpu_rationale_dataset, gpu_data_idx, gpu_winlose, all_correct_total = eval_examples_nohint(args, model, rank, data_queue, tokenizer, gen_length, n_shot_prompts, n_shot_prompts_hint)
        else:
            gpu_rationale_dataset, gpu_data_idx, gpu_winlose, all_correct_total = eval_examples(args, model, rank, data_queue, tokenizer, gen_length, n_shot_prompts, n_shot_prompts_hint)
    else:
        print("device_inference_adastar is not supposed to be called in dev split")
        raise NotImplementedError

    # 여기서 합쳐줌
    all_rationale_dataset = gather_and_merge_list(gpu_rationale_dataset, dst=0)
    all_data_idx = gather_and_merge_list(gpu_data_idx, dst=0)
    all_winlose = gather_and_merge_dicts(gpu_winlose, dst=0)

    return all_rationale_dataset, all_data_idx, all_winlose, all_correct_total

# 이번추론에서 사용할 모델체크포인트 경로를 결정함
# split이 train이고 ckpt_step이 0이면 base_model 반환하는데, 사실상 항상 -1이고, 이전 iter의 체크포인트 반환
def get_ckpt_path(args, ckpt_step=-1):
    '''
    이번추론에서 사용할 모델체크포인트 경로를 결정함
    '''
    model_dir = args.model_dir
    if ckpt_step == -1:
        ckpt_step = args.total_steps    
        
    path = f"{model_dir}/step_{ckpt_step}/lm.pt"

    if args.split == "train" and ckpt_step == 0:
        print(f"Generate train set using base model")
        return "base_model"

    if os.path.exists(path):
        return path
    else: 
        raise FileNotFoundError(f"Model path {path} not found, exiting")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None, help="Config file location")
    parser.add_argument('--rationalize', action='store_true', help="Whether to use rationalization")
    parser.add_argument('--show_hint_prompt', action='store_true', help="Whether a hint prompt will be necessary")
    parser.add_argument("--split", type=str, default="dev", help="Split")
    parser.add_argument("--task", type=str, default="cqa", help="Which dataset to run on")
    parser.add_argument("--ckpt_step", type=int, default=-1, help="Which checkpoint to eval. -1 means the final one")
    parser.add_argument("--exp_iter", type=int, default=-1, help="exp iteration for logging")
    parser.add_argument("--seed", type=int, default=10, help="Random seed for shuffling data (default: 10)")
    parser.add_argument("--log_dir", type=str, default="", help="logging dir")
    parser.add_argument("--cur_total_steps", type=int, required=True, help="current total steps")
    parser.add_argument("--flops_dir", type=str, default="", help="logging dir")

    return parser.parse_args()

def fsdp_main(rank, world_size, args, data_queue):
    # 분산통신 통신그룹 초기화하고
    setup(rank, world_size)
    # 사용할 GPU번호 지정
    torch.cuda.set_device(rank)

    # few_shot 프롬프트 가져오고
    n_shot_prompts = ""
    n_shot_prompts_hint = ""
    prompt_file_path = f"./n_shot_prompts/{args.task}.json"
    prompt_hint_file_path = f"./n_shot_prompts/{args.task}_hint.json"
    with open(prompt_file_path, "r") as f:
        data = json.load(f)
    with open(prompt_hint_file_path, "r") as f:
        data_hint = json.load(f) 
    n_shot_prompts = [item["prompt"] for item in data["n_shot_prompts"]]
    n_shot_prompts_hint = [item["prompt"] for item in data_hint["n_shot_prompts"]]
    n_shot_prompts = "\n".join(n_shot_prompts)
    n_shot_prompts_hint = "\n".join(n_shot_prompts_hint)

    # 이번 추론에서 사용할 체크포인트 경로가져와서 모델과 토크나이저 로드
    ckpt_path = get_ckpt_path(args, args.ckpt_step)
    if ckpt_path != "base_model": 
        dist.barrier()
        model, tokenizer = get_loaded_model_tokenizer(args, ckpt_path, args.model_name, rank, eval=True)
        dist.barrier()
        if rank == 0:
            print(f"[Inference {args.split}] model loaded from {ckpt_path}")
        
    else: 
        if args.base_model_path != None:
            dist.barrier()
            model, tokenizer = get_loaded_model_tokenizer(args, args.base_model_path, args.model_name, rank, eval=True)
            dist.barrier()
            if rank == 0:
                print(f"[Inference {args.split}] base model loaded from {args.base_model_path}")
        else:
            dist.barrier()
            model, tokenizer = get_model_tokenizer(args, args.model_name, rank, eval=True)
            dist.barrier()
            if rank == 0:
                print(f"[Inference {args.split}] base model path == None, using hf base model")

    # fewshot 반영해서 최대 토큰입력길이 늘려줌
    tokenized_prompt = tokenizer(n_shot_prompts, return_tensors="pt")
    prompt_tokenized_len = tokenized_prompt["input_ids"].shape[1]
    tokenized_prompt_hint = tokenizer(n_shot_prompts_hint, return_tensors="pt")
    prompt_tokenized_len_hint = tokenized_prompt_hint["input_ids"].shape[1]
    args.max_length += max(prompt_tokenized_len, prompt_tokenized_len_hint)

    # 채워야할 정답데이터가 총 몇개인지 계산
    effective_batch_size = args.batch_size * args.grad_accumulation * world_size
    args.required_data_num = args.cur_total_steps * effective_batch_size
    if rank == 0:
        print(f"Required data num: {args.required_data_num}")

    # 여기는 학습이 아니라 추론이니까 학습기준으로 설정된 마이크로 배치크기를 추론 배치크기로 변경
    # 추론에는 grad_accum 안하니까 마이크로배치 아니고 그냥 배치임
    args.batch_size = args.test_batch_size

    # svamp/svamp_qwen_adastar_new_square_10/winlose.json, svamp/svamp_qwen_adastar_new_square_10/heap.pkl에서 가져옴
    dataset_train, winlose, heap = get_dataloader_adastar(args, tokenizer, rank=rank, world_size=world_size)

    init_start_event = torch.cuda.Event(enable_timing=True)
    init_end_event = torch.cuda.Event(enable_timing=True)

    if args.split == "train":
        init_start_event.record()
        # 문제풀고 정답데이터셋 채워넣음
        all_rationale_dataset, all_data_idx, all_winlose, all_correct_total = evaluate(args, model, rank, world_size, data_queue, tokenizer, args.gen_length, args.target_save, n_shot_prompts, n_shot_prompts_hint)
        init_end_event.record()
        # svamp/svamp_qwen_adastar_new_square_10/svamp_qwen_adastar_new_square_10_19/elapsed_time_log.json 이런 로그파일 생성
        if rank == 0:
            log_args(f"{args.log_dir}/elapsed_time_log.json", iter=args.exp_iter, log_point="gen_train_data", time=init_start_event.elapsed_time(init_end_event) / 1000)
    else:
        print("device_inference_adastar is not supposed to be called in dev split")
        raise NotImplementedError

    dist.barrier()
    if rank == 0:
        # 힙 알고리즘에 쓸 시도한문제수 m과 정답률 \alpha 계산
        if args.no_hint:
            nohint_correct = int(all_correct_total[0].item())
            nohint_total = int(all_correct_total[1].item())
            
            # 시도한것중에 맞춘횟수
            accuracy = nohint_correct / nohint_total
            hint_accuracy = "_"

            # 총 시도횟수
            total_inference = nohint_total
            total_accuracy = accuracy
        else:
            nohint_correct = int(all_correct_total[0].item())
            hint_correct = int(all_correct_total[1].item())
            nohint_total = int(all_correct_total[2].item())
            hint_total = int(all_correct_total[3].item())

            accuracy = nohint_correct / nohint_total
            hint_accuracy = hint_correct / hint_total

            total_inference = nohint_total
            total_accuracy = accuracy + (1 - accuracy) * hint_accuracy

        # 정답데이터셋에 채울 개수 \beta^t
        num_used_data = args.required_data_num
        # 힙에서 pop할 데이터 수 m\alpha^2 계산하는부분
        num_pop_data = round(total_inference * (total_accuracy ** args.accuracy_power))
        
        # heap에서 m\alpha^2번 뽑고 업데이트해서 다시 heap에 넣어줌. 전역통계도 갱신해줌
        # all_winlose는 이번 iter에서의 통계고, winlose 는 전 iter에서의 전역통계임
        updated_minheap, updated_winlose = update_minheap_winlose_front(heap, winlose, all_winlose, num_pop_data=num_pop_data)

        # all_rationale_dataset이 보통 얼추 args.required_data_num개는 맞지만 정확하진 않고, 재수없어서 depleted되면 한참 모자랄수도 있음.
        # 그래서 최대 args.required_data_num개만큼 끊어주고 저장함
        for new_example, data_idx in zip(all_rationale_dataset[:num_used_data], all_data_idx[:num_used_data]):
            # append모드로 svamp/svamp_qwen_adastar_new_square_10/svamp_qwen_adastar_new_square_10_17/correct_data.txt 이런 파일에 정답 데이터 계속 이어씀
            with open(args.target_save + "/correct_data.txt", 'a+') as new_train_f:
                # 표준출력이 아니라 파일에 쓰게 하고, 다쓰면 2칸 띔
                print(new_example, file=new_train_f, end="\n\n")

            # 정답데이터 쪼개서 문제의 질문부분만 추출
            new_example_no_answer = "A:".join(new_example.split("A:")[:-1])
            # append모드로 svamp/svamp_qwen_adastar_new_square_10/svamp_qwen_adastar_new_square_10_17/train_corr_idx_18.txt 이런 파일에 계속 이어씀
            with open(args.idx_save + f"/{args.split}_corr_idx_{args.exp_iter}.txt", 'a+') as new_idx_f:
                print(f"idx: {data_idx}\n{new_example_no_answer}", file=new_idx_f, end="\n\n")

        # svamp/svamp_qwen_adastar_new_square_10/winlose.json 이렇게 전역통계 저장
        json.dump(updated_winlose, open(args.log_dir + "/winlose.json", "w"))
        # 업데이트된 heap도 svamp/svamp_qwen_adastar_new_square_10/heap.pkl 이렇게 저장
        updated_minheap.save(args.log_dir + "/heap.pkl")


        # svamp/svamp_qwen_adastar_new_square_10/eval_log.json 이렇게 로그 저장
        log_args(f"{args.log_dir}/eval_log.json", iter=args.exp_iter, split=args.split, accuracy=accuracy, hint_accuracy=hint_accuracy)
        # 전역통계를 svamp/svamp_qwen_adastar_new_square_10/svamp_qwen_adastar_new_square_10_17/winlose.json 이렇게 iter별로도 다시 저장해줌
        json.dump(updated_winlose, open(args.target_save + "/winlose.json", "w"))
        # heap도 사람이 보기 좋은형태로 가공한 json 버전을 svamp/svamp_qwen_adastar_new_square_10/svamp_qwen_adastar_new_square_10_17/heap_stats.json 이렇게 저장함
        log_stable_minheap_new(updated_minheap, args.target_save + "/heap_stats.json")
    cleanup()

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)

    args = parse_args()
    print(args)
    split = args.split
    params = json.load(open(args.config))

    args.batch_size = params["batch_size"]
    args.test_batch_size = params["test_batch_size"]
    args.model_name = params["model_name"]
    args.precision = params["precision"]
    args.max_length = params["max_length"]
    args.gen_length = params["gen_length"]
    args.n_shot = params["n_shot"]
    args.self_consistency = params["self_consistency"]
    args.delete_model_after_loading = params["delete_model_after_loading"]
    args.lora = params.get("lora", None)
    args.grad_accumulation = params["grad_accumulation"]
    args.inference_temp = params["inference_temp"]
    args.no_hint = params["no_hint"]
    args.base_model_path = params["base_model_path"]

    args.name = params["name"]
    args.idx_save = params["target_save"] 
    args.target_save = params["target_save"] if split != "dev" else f'{args.task}/new_dev.txt'
    args.model_dir = params["model_dir"]
    args.total_steps = params.get("total_steps", 0)
    
    args.method = params["method"]
    args.accuracy_power = params["accuracy_power"]

    torch.manual_seed(args.seed)

    WORLD_SIZE = torch.cuda.device_count()

    # 토크나이저 불러오고
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    dataset_train, winlose, heap = get_dataloader_adastar(args, tokenizer, rank=0, world_size=WORLD_SIZE)

    with mp.Manager() as manager:
        queue = manager.Queue()
        for item in dataset_train:
            queue.put(item)
        for _ in range(args.test_batch_size * WORLD_SIZE):
            queue.put(None)
        # fsdp_main을 병렬실행하는데 학습하는것도 아니고 사실 DDP쓴다고 함 
        mp.spawn(fsdp_main, args=(WORLD_SIZE, args, queue), nprocs=WORLD_SIZE, join=True)


