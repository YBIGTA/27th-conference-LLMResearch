import os
import argparse
import torch

import torch.distributed as dist
import torch.multiprocessing as mp
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import FullStateDictConfig, StateDictType

import json
from tqdm import tqdm

from utils import get_model_tokenizer, get_optimizer_scheduler_step_based, setup, cleanup, log_args, \
    get_dataloader_STaR, get_loaded_model_tokenizer, merge_flops_logs


def create_labels(input_ids, rank, tokenizer):
    batch_size, seq_len = input_ids.shape # (B, seq_len)
    # input_ids 복제해서 자료형 바꿈
    labels = input_ids.clone().to(torch.int64)

    a_token_id = tokenizer.encode("A", add_special_tokens=False)[0]
    colon_token_id = tokenizer.encode(":", add_special_tokens=False)[0]

    # 배치를 돌면서 각 input_ids에서 마지막 A: 부분 이전을 마스킹함.
    # A: 부분이 없는 데이터는 그냥 -100으로 다 무시
    for i in range(batch_size):
        last_a_colon_position = -1
        for j in range(seq_len - 1):
            if input_ids[i, j] == a_token_id and input_ids[i, j + 1] == colon_token_id:
                last_a_colon_position = j

        if last_a_colon_position != -1:
            labels[i, :last_a_colon_position + 2] = -100 
        else:
            labels[i, :] = -100 
    return labels


def gather_and_merge_logs(tensor, rank, world_size):
    gather_list = [torch.zeros_like(tensor) for _ in range(world_size)]
    dist.all_gather(gather_list, tensor)
    return torch.stack(gather_list)


def train_step_based(args, model, tokenizer, rank, world_size, train_loader, optimizer, scheduler, sampler=None):
    # 훈련모드로 모델 전환하고
    model.train()
    # 실제학습에 필요한건 아니고 로그용 변수라고 함. 분산학습에서는 loss에 이것저것 추가로 담을게 있어서 (2,) 크기의 손실텐서 만들고 GPU로 옮김
    # nccl로 GPU끼리 통신하는데, 결국 loss를 GPU가 들고있어야 통신으로 연산을 하니까...
    ddp_loss = torch.zeros(2).to(rank)
    # 이번 iter내에서 몇번째 스텝인지 변수
    global_step = 0
    # 이번 iter에서 총 몇스텝 학습할지
    # 근데 좀 의문이 드는게, STaR는 iteration마다 스텝이 아니라 데이터셋 개수를 정해두지 않나..?
    # -> 맞는데, 목표한 데이터셋을 못채우는경우가 있어서 현실적으로 딱 떨어지는 total_steps를 쓰는듯
    total_steps = args.total_steps

    # args.batch_size는 GPU에 들어오는 mirco-batch인듯
    effective_batch_size = args.batch_size * args.grad_accum * world_size
    # 이번 iter에서 필요한 총 데이터 개수
    args.required_data_num = total_steps * effective_batch_size

    # 사용된 데이터를 체크하기위한 집합, 딕셔너리
    used_indices = set()
    step_indices = {}

    dataloader_iterator = iter(train_loader)
    # tqdm은 for문으로 쓸수도 있고, .update()로 수동으로 쓸수도 있다고 함.
    progress_bar = tqdm(
        total=total_steps,
        desc=f"Training [Rank {rank}]", # 설명 문자열
        position=rank,
        leave=False,
        disable=(rank != 0), # 분산학습에서 rank 0일때만 출력
    )

    # 로그출력할 간격 설정
    args.log_interval = max(1, round(total_steps // args.log_divisor))

    # 학습 시작
    while global_step < total_steps:
        # 이번 step에서 사용한 모든 데이터 체크용 리스트
        step_batch_indices = []

        # grad_accum번 돌면서 그라디언트 누적
        for idx in range(args.grad_accum):

            # trainset을 순회함. 
            # 아까 봤듯이, 실제 알고리즘과 달리 여기서는 한 iter을 돌때 정답데이터셋 한 epoch 돌았냐가 아니라 
            # 현실적인 이유로 total_steps를 쓰기때문에, trainset을 다 돌았어도 total_steps 미달이면 다시 돔
            try:
                data = next(dataloader_iterator)
            except StopIteration:
                dataloader_iterator = iter(train_loader)
                if sampler:
                    sampler.set_epoch(global_step // len(train_loader))
                data = next(dataloader_iterator)

            # 마이크로배치 인덱스들 기록. 여러개로 된 리스트형태이므로 append대신 extend 씀.
            batch_indices = data["idx"]
            step_batch_indices.extend(batch_indices)

            input_ids = data["input_ids"]
            attention_mask = data["attention_mask"]
            # self-supervised 방식말고 A: 뒷부분만 학습시킬거라 앞부분은 -100으로 마스킹
            labels = create_labels(input_ids, rank, tokenizer)

            # GPU에 텐서들 옮기고
            input_ids, attention_mask, labels = (
                input_ids.to(rank),
                attention_mask.to(rank),
                labels.to(rank),
            )

            # AutoModelForCausalLM 같은데서 불러온 모델은 추상화가 되어있어서 criterion 할 필요없음
            # labels에 목표 넣어주면 loss구할때 자동으로 한칸 shift해서 계산해준다고 함.
            output = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            # gradient accumulation 쓰니까 나눠주고
            loss = output.loss / args.grad_accum
            # 마이크로배치의 gradient는 자동으로 누적되고있음
            loss.backward()

            # rank별 flops 로깅.  svamp/svamp_qwen_adastar_new_square_10/svamp_qwen_adastar_new_square_10_12/flops_log_1.json 이런식
            flops_log_file = f"{args.log_dir}/flops_log_{rank}.json"
            log_args(flops_log_file, iter=args.exp_iter, idx=global_step, split="train",
                    batch=args.batch_size, grad_accum=args.grad_accum,
                    context_len=args.batch_size*input_ids.size(1))
            
            # ddp_loss에 loss와 데이터 수 누적. 한 배치 단위도 아니고 한 마이크로배치 단위도 아님
            ddp_loss[0] += loss.item() * args.grad_accum
            ddp_loss[1] += len(input_ids)

        # 이번 step에서 사용한 데이터 인덱스 수합.
        # 마이크로배치 인덱스 Tensor형태면 int 형으로 바꿔주고
        step_batch_indices = [int(idx) if isinstance(idx, torch.Tensor) else idx for idx in step_batch_indices]
        # 리스트형태라서 Tensor로 바꾸고 GPU로 옮겨줌
        indices_tensor = torch.tensor(step_batch_indices, device=rank)
        # indices_tensor랑 장치, 자료형까지 완전히 동일한 zeros 텐서 world_size 개수만큼 만들기
        gathered_indices = [torch.zeros_like(indices_tensor) for _ in range(world_size)]
        # 분산통신하는데, gathered_indices에 각 rank의 indices_tensor가 채워짐
        dist.all_gather(gathered_indices, indices_tensor)

        # 수합한 인덱스들 flat하게 저장
        all_step_indices = []
        for gathered in gathered_indices:
            all_step_indices.extend(gathered.cpu().tolist())

        # 학습한 데이터 인덱스확인용 집합 업데이트 
        used_indices.update(all_step_indices)
        # 스텝별로 사용한 데이터 인덱스 딕셔너리에 추가
        step_indices[global_step] = all_step_indices

        # rank가 0이고 로깅 쿨 돌았으면 어떤 데이터를 썼는지에 대한 로그 기록
        if rank == 0 and ((global_step + 1) % args.log_interval == 0 or (global_step + 1) == total_steps):
            log_entry = {
                "iter": args.exp_iter,
                "step": global_step,
                "batch_indices": all_step_indices,
                "total_unique_indices": len(used_indices),
                "batch_size": len(all_step_indices)
            }

            os.makedirs(args.log_dir, exist_ok=True)
            index_log_file = f"{args.log_dir}/training_indices_log.json"

            try:
                with open(index_log_file, 'r') as f:
                    logs = json.load(f)
            except (FileNotFoundError, json.JSONDecodeError):
                logs = []

            logs.append(log_entry)
            with open(index_log_file, 'w') as f:
                json.dump(logs, f, indent=2)

        # reduce-scatter는 FSDP 래퍼가 순전파나 역전파때 알아서 해준다고 함.
        # 사실 역전파때 reduce-scatter하는것뿐만 아니라 순전파때도 all-gather해야됐는데 래퍼가 알아서 해주고있었던거임
        optimizer.step()
        optimizer.zero_grad()
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        # 모든 스텝 다돌아서 이번 iter 끝났으면, 모델 저장하고 필요하면 오래된모델은 삭제
        if (global_step + 1) == total_steps:
            save_consolidated_model(args, model, args.total_steps, args.model_dir, rank)

        # 로깅할 쿨 됐거나 모든스텝 다 돌았으면 loss같은 훈련 지표 tqdm바에 띄우고 로그도 기록
        if (global_step + 1) % args.log_interval == 0 or (global_step + 1) == total_steps:
            # ddp_loss를 all-reduce 하는데, 평균하는게 아니라 모든요소 다 더하는방식으로 함.
            dist.all_reduce(ddp_loss, op=dist.ReduceOp.SUM)
            # 누적 ddp_loss를 데이터 수로 나눠서 평균 loss 계산
            avg_loss = (ddp_loss[0] / ddp_loss[1]).item()

            if rank == 0:
                # Train:  40%|████      | 2/5 [00:00<00:00, ...it/s, loss=0.731, lr=6.0e-05] 이런식으로 tqdm바에도 훈련 지표 업데이트해서 보여줌
                progress_bar.set_postfix({
                    "Loss": avg_loss,
                    "LR": current_lr,
                    "Unique Indices": len(used_indices)
                })

                # svamp/svamp_qwen_adastar_new_square_10/svamp_qwen_adastar_new_square_10_22/train_log.json 이런식으로 로그경로 설정하고
                trainfile = f"{args.log_dir}/train_log.json"
                # 이번 로그 내용 채운뒤
                log_entry = {
                    "iter": args.exp_iter,
                    "step": global_step,
                    "loss": avg_loss,
                    "learning_rate": current_lr,
                    "unique_indices": len(used_indices)
                }

                try:
                    with open(trainfile, 'r') as f:
                        train_logs = json.load(f)
                except (FileNotFoundError, json.JSONDecodeError):
                    train_logs = []

                # iter의 로그에 덮어씀
                train_logs.append(log_entry)
                with open(trainfile, 'w') as f:
                    json.dump(train_logs, f, indent=2)

            # rank가 0인 애가 로그쓰느라 늦으니까 여기서 같이 타이밍 맞춰줌
            dist.barrier()
            # ddp_loss는 로깅용이므로 로깅했으니 다시 초기화
            ddp_loss = torch.zeros(2).to(rank)

        # 한스텝 돌았으면 step 하나 늘리고
        global_step += 1
        # tqdm바도 업데이트
        progress_bar.update(1)

    # 모든 step 다 돌고 이번 iter 훈련 끝났으면, rank가 0일때 사용한 데이터 최종본 로그작성
    if rank == 0:
        final_stats = {
            "iter": args.exp_iter,
            "total_steps": total_steps,
            "total_unique_indices_used": len(used_indices),
            "unique_indices": sorted(list(used_indices)),
            "indices_per_step": {str(k): v for k, v in step_indices.items()}
        }
        # svamp/svamp_qwen_adastar_new_square_10/svamp_qwen_adastar_new_square_10_22/final_indices_stats.json 이런식으로 저장
        with open(f"{args.log_dir}/final_indices_stats.json", 'w') as f:
            json.dump(final_stats, f, indent=2)

    # tqdm바도 종료
    progress_bar.close()


# shard 합쳐서 FullStateDict만들어서 모델 저장하고, delete_path있으면 오래된 모델도 삭제
def save_consolidated_model(args, model, step, path, rank):  
    '''
    shard 합쳐서 FullStateDict만들어서 모델 저장하고, delete_path있으면 오래된 모델도 삭제
    '''
    # path는 checkpoints/svamp_qwen_adastar_new_square_10_23/ 이런식으로 들어옴
    assert path
    if rank == 0:
        # checkpoints/svamp_qwen_adastar_new_square_10_23/step_2208 이런식으로 체크포인트 만들 디렉토리 생성
        step_path = os.path.join(path, f"step_{step}")
        os.makedirs(step_path, exist_ok=True)

    # state dict는 모델의 구조별 가중치를 담은 딕셔너리라고 함. 분산학습중이라  GPU별로 쪼개진 shard들을 합쳐 FULL_STATE_DICT를 만들어야함.
    # 분산학습중이라  GPU별로 쪼개진 shard들을 합쳐 FULL_STATE_DICT를 만들어야하는데, 합칠때 더 용량이 큰 CPU RAM 쓰고, 저장은 rank0만 하게 설정
    cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)

    # FullStateDict 만들때 여러 GPU들의 도움이 필요할수 있어서 타이밍 꼬이지않게 다같이 진입하도록 함
    dist.barrier()
    # 이 with 안에서는 model.state_dict()로 model의 state dict 뽑을때 각 shard들 모아서 FULL_STATE_DICT로 만들고, 그때 아까 설정해둔 cfg를 보고 하라는뜻.
    with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, cfg):
        # 여기서 실제로 state dict를 뽑음
        full_state_dict = model.state_dict()
        # rank가 0일때만 저장하는데,
        if rank == 0:
            # 만약 lora면 loar_키만 저장
            if args.lora:  
                lora_state_dict = {k: v for k, v in full_state_dict.items() if "lora_" in k}
                full_state_dict = lora_state_dict

            # 아까 생성한 step_path 디렉토리에 lm.pt로 state dict 저장
            torch.save(full_state_dict, f"{step_path}/lm.pt")
            print(f"model saved at {step_path}/lm.pt")
            # 만약 delete_path 가 주어졌으면, 그거보고 오래된 체크포인트 삭제
            if args.delete_path is not None:
                ckpt_path=args.delete_path
                if os.path.exists(ckpt_path):
                    os.remove(ckpt_path)
                    print(f"Model file {ckpt_path} deleted successfully.")
                else:
                    print(f"Model file {ckpt_path} not found.")
    # FullStateDict 뽑는거 끝났으니까 이후에 타이밍 꼬이지 않게 다같이 나가도록 함
    dist.barrier()


def fsdp_main(rank, world_size, args):
    # 분산학습시 통신을 위해 프로세스 그룹 초기화
    setup(rank, world_size)
    # 각 프로세스가 디폴트로 사용할 GPU를 지정해버림
    torch.cuda.set_device(rank) 

    # 직전 모델 체크포인트 경로가 없는데
    if args.model_path == None: 
        # 베이스모델은 있으면 베이스모델 쓰고
        if args.base_model_path != None:
            model, tokenizer = get_loaded_model_tokenizer(args, args.base_model_path, args.model_name, rank)
            # rank 0에서만 로그출력
            if rank == 0:
                print(f"[Train] base model loaded from {args.base_model_path}")
        # 베이스모델도 없으면 허깅페이스에서 다운로드
        else:
            model, tokenizer = get_model_tokenizer(args, args.model_name, rank)
            if rank == 0:
                print("[Train] base model path == None, using hf model")
    # 체크포인트 경로가 있으면 모델로 그 체크포인트 사용
    else:
        model, tokenizer = get_loaded_model_tokenizer(args, args.model_path, args.model_name, rank)
        if rank == 0:
            print(f"[Train] model loaded from {args.model_path}")

    # .index 경로를 보고 .pt로 텐서화된 정답데이터셋 경로 인자로 저장
    with open(args.data_dir, 'r') as f:
        pt_file_path = f.read().strip() 
        args.dataset_path = pt_file_path
    # 텐서화된 정답 데이터셋보고 train_loader 불러옴
    # train_loader는 진짜 그거 맞고, sampler_train은 분산학습때 싱크로맞출려고 train_loader랑 항상 따라다니는애
    train_loader, sampler_train = get_dataloader_STaR(args, tokenizer, rank, world_size)

    # GPU는 비동기적으로 연산을 해서 정확히 GPU 계산시간잴려면 time.time말고 이렇게 해야한다고 함
    init_start_event = torch.cuda.Event(enable_timing=True)
    init_end_event = torch.cuda.Event(enable_timing=True)

    # 옵티마이저랑 스케쥴러 받아오고
    optimizer, scheduler = get_optimizer_scheduler_step_based(args, model, train_loader)
    init_start_event.record()

    # 실제 훈련은 여기서 돌림
    train_step_based(args, model, tokenizer, rank, world_size, train_loader, optimizer, scheduler,
                     sampler=sampler_train)

    # GPU 연산시간 측정
    init_end_event.record()

    # rank 0일때만
    if rank == 0:
        # GPU 연산시간 로그 출력하고
        print(f"CUDA event elapsed time: {init_start_event.elapsed_time(init_end_event) / 1000}sec")
        # json로그에도 기록해준다고 함
        log_args(f"{args.log_dir}/../elapsed_time_log.json", iter=args.exp_iter, log_point="train",
                 time=init_start_event.elapsed_time(init_end_event) / 1000)
        # rank별로 flops 계산한 로그파일 합친 로그파일 만든다고 함
        merge_flops_logs(args)

    # 분산학습통신을 위한 프로세스 그룹 종료
    cleanup()


if __name__ == '__main__':
    # CLI로 들어온 인자들 파싱해주고
    parser = argparse.ArgumentParser(description='FSDP Finetune')
    parser.add_argument("--config", type=str, default="configs/base_fsdp.json", help="Config file location")
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument("--seed", type=int, default=10, help="Random seed for shuffling data (default: 10)")

    parser.add_argument("--exp_iter", type=int, default=-1, help="exp iteration for logging")
    parser.add_argument("--log_dir", type=str, default="", help="logging dir")
    parser.add_argument("--flops_dir", type=str, default="", help="logging dir")
    parser.add_argument("--data_dir", type=str, default="", help="data dir")
    parser.add_argument("--model_path", type=str, default=None, help="model_path")
    parser.add_argument("--delete_path", type=str, default=None, help="delete_path")

    args = parser.parse_args()
    torch.manual_seed(args.seed)

    # config 파일읽어서 args 보충해줌
    # configs/svamp_qwen_adastar_new_square_10/svamp_qwen_adastar_new_square_10_23.json 이런게 들어옴
    params = json.load(open(args.config))

    args.batch_size = params["batch_size"]
    args.test_batch_size = params["test_batch_size"]
    args.grad_accum = params["grad_accumulation"]
    args.gen_length = params["gen_length"]
    args.ckpt_save = params["model_dir"]
    args.log_divisor = params["log_divisor"]
    args.lr = params["lr"]
    args.weight_decay = params["weight_decay"]
    args.warm_up_steps = params["warm_up_steps"] 
    args.scheduler = params["scheduler"]  
    args.optimizer = params["optimizer"] 
    args.precision = params["precision"] 
    args.model_name = params["model_name"]
    args.max_length = params["max_length"]
    args.task = params["task"]
    args.lora = params.get("lora", None)
    args.accumulate = params["accumulate"] 
    args.split = 'ft'
    args.base_model_path = params["base_model_path"]
    
    args.total_steps = params["total_steps"]
    args.name = params["name"]
    args.model_dir = params["model_dir"]
    args.method = params["method"]
    # CUDA GPU 개수 감지하고 변수로 할당
    WORLD_SIZE = torch.cuda.device_count()
    # 멀티프로세싱으로 fsdp_main를 돌림.
    # 첫번째 인자로 돌릴 함수 주소를 넣어주면, 그 함수 돌릴때 rank정보와 두번째 인자로 함수에 넣어줄 인자를 같이 넣어줌
    # nprocs는 여기서 돌릴 총 프로세스 수이고, join은 모든프로세스가 끝날때까지 메인프로세스가 기다릴지 여부임.
    mp.spawn(fsdp_main,
             args=(WORLD_SIZE, args),
             nprocs=WORLD_SIZE,
             join=True)
