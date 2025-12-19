import os
import sys
import json
import shutil
import argparse
import torch
import time
import glob
from utils import log_args


def record_folder(cur_iter):
    '''
    {task}/{experiment_name}/{experiment_name}_{cur_iter} 형태의 경로 반환
    '''
    return f"{task}/{experiment_name}/{experiment_name}_{cur_iter}"


# 입력받은 CLI에서 인자 파싱하고 args에 담아서 반환하는 함수
def parse_args():
    '''
    입력받은 CLI에서 인자 파싱하고 디폴트도 채워서 args에 담아 반환하는 함수
    '''
    parser = argparse.ArgumentParser()

    parser.add_argument("--steady_grow", action='store_true', help="Whether to use a fixed number of epochs")
    parser.add_argument("--add_steps", type=float, default=20., help="Steps to add each iteration")

    parser.add_argument("--start_steps", type=float, default=40, help="Steps for the first iteration")
    parser.add_argument("--exponential_grow", type=bool, default=True, help="Whether to use a fixed number of epochs")
    parser.add_argument("--grow_steps", type=float, default=1.2, help="Steps to add each iteration")

    parser.add_argument("--p_rationalization", type=float, default=1., help="Percent of wrong examples to rationalize")
    parser.add_argument("--p_show_hint_save", type=float, default=0., help="Percent of rationalization hints to save")
    parser.add_argument('--rationalize', action="store_true", default=True, help="Whether to use rationalization")

    parser.add_argument("--start_iter", type=int, default=1, help="Starting iteration")
    parser.add_argument("--n_iters", type=int, default=46, help="Upper limit on outer loop iterations")

    parser.add_argument("--copy_n", type=int, default=0, help="Number of files to copy each iteration")

    parser.add_argument("--config", type=str, required=True, help="Base config file location")
    parser.add_argument("--seed", type=int, required=True, help="Random seed")
    parser.add_argument("--method", type=str, default="vanilla", help="training method (vanilla, dpo)")

    parser.add_argument('--dry_run', action='store_true', help="Whether to do a quick run to visualize output")
    parser.add_argument('--skip_eval', action='store_true', help="Whether to skip evaluation (e.g. arithmetic)")

    args = parser.parse_args()
    return args


def gen_train():
    if args.method == "adastar_new_square":
        train_cmd = f"python3 device_inference_adastar_new_square.py --config={prev_config} --split=train --seed={args.seed}"
        train_cmd += f" --task={task}"
        train_cmd += " --rationalize"
        train_cmd += f" --log_dir={task}/{experiment_name} --exp_iter={cur_iter}"
        train_cmd += f" --cur_total_steps={get_n_steps()}"

    elif args.method == "adastar_new":
        train_cmd = f"python3 device_inference_adastar_new.py --config={prev_config} --split=train --seed={args.seed}"
        train_cmd += f" --task={task}"
        train_cmd += " --rationalize"
        train_cmd += f" --log_dir={task}/{experiment_name} --exp_iter={cur_iter}"
        train_cmd += f" --cur_total_steps={get_n_steps()}"

    print(f"Generating training set {cur_iter} using model {cur_iter - 1}: {train_cmd}")
    if not args.dry_run and (cur_iter >= args.start_iter):
        if args.method in {"adastar_new_square", "adastar_new"} and (cur_iter == 1) and os.path.exists(
                record_folder(0) + f"/correct_data.txt"):
            print("First file cached")
        else:
            os.system(train_cmd)


# 만들어놓은 정답데이터셋 파이토치 텐서화해서 .pt파일 만들고 위치는 .index파일에 저장
def gen_records():
    '''
    만들어놓은 정답데이터셋을 PyTorch 텐서화해서 .pt파일 만들고 위치는 .index파일에 저장
    '''
    # gsm8k/gsm8k_adastar_new_square_10/gsm8k_adastar_new_square_10_7/ 이런 이전 iter의 디렉토리로 들어가서
    # gen_train()에서 만들어놓은 correct_data.txt보고 토큰화한다음에 PyTorch 텐서로 변환하는 명령어
    # svamp/svamp_qwen_adastar_new_square_10/svamp_qwen_adastar_new_square_10_7.pt 이런식으로 저장되는듯
    gen_cmd = f'python3 create_finetune_tfrecords.py {record_folder(cur_iter - 1)} {record_folder(cur_iter - 1)}  --model_name={args.model_name} --seed={args.seed}'
    gen_cmd += f' --max-length={args.max_length}'
    gen_cmd += f' --idx_save={record_folder(cur_iter - 1)}'
    gen_cmd += f' --split=train'
    gen_cmd += f' --exp_iter={cur_iter}'

    print(f"Creating records for finetuning {cur_iter}: {gen_cmd}")
    # dry_run 모드가 아니면 명령어 실행
    if not args.dry_run and (cur_iter >= args.start_iter):
        if args.method in {"adastar_new_square", "adastar_new"}:
            os.system(gen_cmd)

    train_set = f"{experiment_name}/{exp_iteration}.index"

    # 파일에 .pt파일 위치 로그 한줄 추가
    if args.method in {"adastar_new_square", "adastar_new"}:
        with open(f"data/{train_set}", "w") as new_data_file:
            new_data_file.write(f"{record_folder(cur_iter - 1)}.pt")
    return
 

# 이번 iteration에서 full batch의 크기 계산
def get_n_steps():
    '''
    특정 iteration에서 full batch의 크기 계산
    '''
    # 선형이면 진짜 선형대로 커지고
    if args.steady_grow:
        return int(args.start_steps + args.add_steps * (cur_iter - 1))
    # 논문에서처럼 기본적 방식으로는 지수적으로 커짐
    elif args.exponential_grow:
        return int(args.start_steps * (args.grow_steps ** (cur_iter - 1)))
    # 이건 이젠 안쓰는 폐기된 케이스같음. 신경 X
    else:
        total_count = 0
        for cur_file in sorted(os.listdir(record_folder(cur_iter - 1)),key=lambda x: int(x.split('.')[0].split("_")[-1])):
            with open(f"{record_folder(cur_iter - 1)}/{cur_file}", encoding='utf-8') as train_file:
                train_file_text = train_file.read()
                total_count += len(train_file_text.split("\n\n"))
                print(len(train_file_text.split("\n\n")))
        train_epochs = args.base_epochs + args.add_epochs * (cur_iter - 1)
        cur_steps = int(total_count * train_epochs // (args.batch_size * args.grad_accumulation * torch.cuda.device_count()))
        return cur_steps


# prev_config 복사해서 이번 iter에 맞게 config 업데이트하고 저장
# configs/svamp_qwen_adastar_new_square_10/svamp_qwen_adastar_new_square_10_23.json 이런식의 경로 반환
def gen_config():
    print(f"Creating new config file {cur_iter}")
    config_name = f'configs/{experiment_name}/{exp_iteration}.json'
    os.makedirs(record_folder(cur_iter), exist_ok=True)
    with open(prev_config, encoding='utf-8') as base_json_file:
        new_json = json.load(base_json_file)
        new_json["model_dir"] = f"{args.base_model_location}/{exp_iteration}"
        new_json["target_save"] = record_folder(cur_iter)
        new_json["total_steps"] = get_n_steps()
        new_json["name"] = exp_iteration
        new_json["p_rationalization"] = args.p_rationalization
        new_json["grad_accumulation"] = args.grad_accumulation
    with open(config_name, "w", encoding='utf-8') as new_json_file:
        json.dump(new_json, new_json_file, indent=2)
    os.makedirs(new_json["model_dir"], exist_ok=True)
    return config_name


# 모델 훈련하고 전전 모델 체크포인트는 삭제하는 함수
def train_model():
    '''
    모델 훈련하고 전전 모델 체크포인트는 삭제하는 함수
    '''
    if args.method in {"adastar_new_square", "adastar_new"}:
        model_cmd = f"python device_train.py --config {config_name} --exp_iter={cur_iter} --seed={args.seed} --log_dir={record_folder(cur_iter - 1)} --data_dir=data/{experiment_name}/{exp_iteration}.index"
        # log_gen에 2개 이상이 들어있을때, 즉 iteration=3 부터는 메모리 절약 위해 전전모델 삭제
        if len(log_gen) >1:  
            delete_path = f"{args.base_model_location}/{experiment_name}_{cur_iter - 2}/step_{log_gen[0]}/lm.pt"
            model_cmd += f" --delete_path={delete_path}"
        # accumulate 할거면 직전모델 경로 추가
        if args.accumulate == True and cur_iter != 1:
            # checkpoints/svamp_qwen_adastar_new_square_10_23/step_2208/lm.pt 처럼 생김
            model_path = f"{args.base_model_location}/{experiment_name}_{cur_iter - 1}/step_{log_gen[-1]}/lm.pt"
            model_cmd += f" --model_path={model_path}"

    print(f"Train model {cur_iter}: {model_cmd}")
    # dry_run 아니면 위의 명령어 실행
    if not args.dry_run and (cur_iter >= args.start_iter):
        os.system(model_cmd)


def eval_model():
    eval_cmd = f"python3 device_inference.py --config={config_name} --split=dev --seed={args.seed}"
    eval_cmd += f" --task={task} "
    eval_cmd += f" --log_dir={task}/{experiment_name} --exp_iter={cur_iter}"
    eval_cmd += f" --flops_dir={record_folder(cur_iter-1)}"

    print(f"Eval model {cur_iter}: {eval_cmd}")
    if not args.dry_run and (cur_iter >= args.start_iter) and not args.skip_eval:
        os.system(eval_cmd)


def copy_files():
    all_files = sorted(os.listdir(record_folder(cur_iter - 1)), key=lambda x: int(x.split('.')[0].split("_")[-1]))
    relevant_files = all_files[-args.copy_n:]
    for cur_file in relevant_files:
        shutil.copy(f"{record_folder(cur_iter - 1)}/{cur_file}", record_folder(cur_iter))


# base.json을 보고 args도 채워주고, base.json에도 설정값을 더 채워넣어주는 함수
def make_first_config():
    '''
    args와 base.json간에 없는부분을 서로 일부채워주는 함수
    '''
    # 바닐라가 아니면 method_json파일 불러오고, 바닐라면 아무것도안함
    if args.method != "vanilla":
        with open(f'configs_method/{args.method}.json', 'r') as method_json_file:
            # 파일스트림객체를 딕셔너리자료형으로 로드
            method_json = json.load(method_json_file)
    else:
        method_json = {}

    with open(prev_config, encoding='utf-8') as base_json_file:
        new_json = json.load(base_json_file)

        # prev_config에서(여기선 최초이므로 base.json) 설정값들 보고 args에 채워주고
        args.batch_size = new_json["batch_size"]
        args.grad_accumulation = new_json["grad_accumulation"]
        args.model_name = new_json["model_name"]
        args.n_shot = new_json["n_shot"]
        args.base_model_location = new_json["model_dir"]
        args.gen_length = new_json["gen_length"]
        args.delete_model_after_loading = new_json["delete_model_after_loading"]
        args.accumulate = new_json["accumulate"]
        args.max_length = new_json["max_length"]
        args.warm_up_steps = new_json["warm_up_steps"]
        args.task = new_json["task"]
        global task
        task = args.task

        # gsm8k/gsm8k_adastar_new_square_10/gsm8k_adastar_new_square_10_0/ 이렇게 최초 레코드 디렉토리 만들고
        os.makedirs(record_folder(0), exist_ok=True)
        # 이번엔 args에서 값 꺼내서 new_json에 넣어줌
        new_json["p_rationalization"] = args.p_rationalization
        new_json["target_save"] = record_folder(0)
        new_json["name"] = f"{experiment_name}_0"
        new_json["method"] = args.method
        # method config 순회하면서 설정값을 보고 base.json에 넣어줌
        for key, value in method_json.items():
            new_json[key] = value

    with open(prev_config, "w", encoding='utf-8') as base_json_file:
        # 여기서 실질적으로 base.json에 덮어씀
        json.dump(new_json, base_json_file, indent=2)
    return new_json


# 전에 몇 iteration까지 했는지 찾는 함수
def find_last_completed_iteration(experiment_name, n_iters):
    '''
    로그파일 "{task}/{experiment_name}/eval_log.json" 를 뒤져서 전에 몇 iteration까지 했는지 찾는 함수
    '''
    # 로그파일 경로를 만들고
    log_file_path = f"{task}/{experiment_name}/eval_log.json"

    # 만약 로그파일이 만들어진적 없으면 훈련한적 없는것이므로 재시작 iteration은 0
    if not os.path.exists(log_file_path):
        return 0, None

    try:
        # 로그파일을 읽어서
        with open(log_file_path, 'r') as log_file:
            logs = json.load(log_file)

            # 만약 로그파일이 list형식이면 정상적으로 기록된 케이스이고
            if isinstance(logs, list):
                completed_iters = {}

                # 로그 리스트를 순회하는데
                for entry in logs:
                    # 딕셔너리 형식이 아니거나 'iter' key가 없는 비정상 경우는 패스 
                    if not isinstance(entry, dict) or 'iter' not in entry:
                        continue

                    iter_num = entry['iter']
                    split = entry.get('split')

                    # completed_iters 딕셔너리에 각 iter의 완료상태 기록
                    if iter_num not in completed_iters:
                        completed_iters[iter_num] = {'train': False, 'dev': False}
                    if split == 'train':
                        completed_iters[iter_num]['train'] = True
                    elif split == 'dev':
                        completed_iters[iter_num]['dev'] = True

                # completed_iters에 train과 dev가 둘다 들어있어야 완료한 iter로 기록
                fully_completed_iters = [
                    iter_num for iter_num, status in completed_iters.items()
                    if status['train'] and status['dev']
                ]

                # fully_completed_iters가 비어있는 엣지케이스는 iter 0으로 방어
                if not fully_completed_iters:
                    return 0, None

                # fully_completed_iters에서 마지막 iter 숫자 추출하고 반환
                last_full_iter = max(fully_completed_iters)
                return last_full_iter, None

            # 쓸일이 있는진 모르겠는데 로그파일이 딕셔너리형태일때는 1을 빼서 iter 반환
            elif isinstance(logs, dict) and 'iter' in logs:
                iter_num = logs['iter']
                return iter_num - 1, None

    # 로그파일이 잘못된경우엔 0 반환
    except json.JSONDecodeError:
        return 0, None

    # 예상치 못한 케이스 방어
    return 0, None


if __name__ == "__main__":
    # CLI에서 argument들 파싱받고
    args = parse_args()
    print(args)
    # python iteration_train.py --config=configs/gsm8k.json --method=adastar_new_square --seed=10 이런식 입력이 들어오면
    # gsm8k_adastar_new_square_10 이런식으로 변형하고
    experiment_name = "_".join([args.config.split("/")[-1].split(".")[0], args.method, str(args.seed)])
    # 안전한 실행을 위해 숫자, 알파벳, _ 빼고는 다 필터링
    experiment_name = ''.join(ch for ch in experiment_name if ch.isalnum() or ch == "_")

    # configs/ 디렉토리에 {experiment_name}으로 디렉토리 생성하고
    os.makedirs(f"configs/{experiment_name}", exist_ok=True)
    # 해당 config 파일 복사해서 configs/{experiment_name}/ 디렉토리 base.json만들고 넣음
    shutil.copy(args.config, f"configs/{experiment_name}/base.json")

    # prev_config에 base.json의 경로를 넣어주고
    prev_config = f"configs/{experiment_name}/base.json"
    # CLI 디폴트인자등을 포함한 실전 config.json파일을 받음
    new_json = make_first_config()
    task = args.task

    # 전에 몇 iter까지 했는지 검색함. 
    # restart_point랑 args.n_iters는 지금은 없어도 되는 아무 기능없는 과거의 흔적
    last_completed_iter, restart_point = find_last_completed_iteration(experiment_name, args.n_iters)

    args.start_iter = last_completed_iter + 1
    # 직전 iteration, 전전 iteration의 full batch 크기를 담아놓는 리스트. train_model()에서 쓴다고 함
    log_gen =[]

    # 만약 이전에 이미 돌린적이 있다면
    if last_completed_iter > 0:
        # 마지막으로 돌린 iter의 config 파일 경로를 설정하고
        prev_config = f'configs/{experiment_name}/{experiment_name}_{last_completed_iter}.json'
        # 경로가 만약 없다면 버그이므로 에러로그 출력하고 base.json으로 fallback함.
        if not os.path.exists(prev_config):
            print(f"Warning: Could not find config file for last completed iteration: {prev_config}")
            print("Falling back to base config")
            prev_config = f"configs/{experiment_name}/base.json"
        # 경로가 있다면 해당 iter부터 재훈련한다고 로그출력
        else:
            print(f"Starting iteration {args.start_iter} using config from iteration {last_completed_iter}")

        # 직전 iter의 flops, indices같은 로그파일들 모두 지워주기
        flops_log_files = glob.glob(record_folder(last_completed_iter) + f"/flops_log*.json")
        for file in flops_log_files:
            os.remove(file)
        indice_log_files = glob.glob(record_folder(last_completed_iter) + f"/*indices_log.json")
        for file in indice_log_files:
            os.remove(file)
        file_path=record_folder(last_completed_iter) + f"/final_indices_stats.json"
        if os.path.exists(file_path):
            os.remove(file_path)
        file_path=record_folder(last_completed_iter) + f"/train_log.json"
        if os.path.exists(file_path):
            os.remove(file_path)
        # 전 iteration의 full batch, 전전 iteration의 full batch 크기로 log_gen을 채워줌
        for cur_iter in range(args.start_iter - 2,args.start_iter):
            exp_iteration = f"{experiment_name}_{cur_iter}"
            log_gen.append(get_n_steps())

    os.makedirs(f'data/{experiment_name}', exist_ok=True)
    os.makedirs(f'{task}/{experiment_name}', exist_ok=True)


    # 재개 iter부터 훈련루프 재개
    for cur_iter in range(args.start_iter, args.n_iters + 1):
        # svamp_qwen_adastar_new_square_10_7 이런식
        exp_iteration = f"{experiment_name}_{cur_iter}"

        # train set 풀면서 AdaSTaR 방법론대로 이번 iter에서 학습에 쓸 정답데이터셋 채우기
        gen_train()

        start_time = time.time()
        # 정답데이터셋 텐서화
        gen_records()
        # 텐서화 경과시간 측정
        elapsed_record = time.time() - start_time
        # 경과시간 로그 기록
        file = f"{task}/{experiment_name}/elapsed_time_log.json"
        log_args(file, iter=cur_iter, log_point="tfrecord", time=elapsed_record)

        # prev_config 복사해서 이번 iter에 맞게 config 업데이트하고 저장
        config_name = gen_config()
        # 텐서화된 정답데이터셋으로 모델 훈련
        train_model()
        # dev set으로 모델 평가
        eval_model()

        # prev_config랑 log_gen 갱신
        prev_config = config_name
        log_gen.append(get_n_steps())
        if len(log_gen) > 2:
            log_gen.pop(0)

        # 이전 iter의 correct_data에서 n개 복사해오는 데이터 재활용코드. 이젠 거의 안쓰는 폐기코드인듯
        if args.copy_n > 0:
            copy_files()
