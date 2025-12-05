import os
import argparse
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import json
from tqdm import tqdm
import gc

from utils.utils import get_model_tokenizer, get_dataloader, get_test_loader_gsmsymb, get_optimizer_scheduler, setup, cleanup, save_model, save_lora_model
from utils.utils_test import create_labels, test, log_args

if __name__ == '__main__':
    # Training settings
    parser = argparse.ArgumentParser(description='FSDP Finetune')
    parser.add_argument("--config", type=str, required=True, help="Config file location")
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, required=True,help='random seed')
    parser.add_argument('--noise_method', type=bool, default=False, help='train noise method')
    parser.add_argument('--method', type=str, default="vanilla", help='train method default: vanilla')

    args = parser.parse_args()
    torch.manual_seed(args.seed)

    params = json.load(open(args.config))

    args.batch_size = params["batch_size"]
    args.test_batch_size = params["test_batch_size"]
    args.grad_accum = params["grad_accumulation"]
    args.gen_length = params["gen_length"]
    args.eval_save = params["eval_save"]
    args.log_save = params["log_save"]
    args.ckpt_save = params["ckpt_save"]
    args.log_divisor = params["log_divisor"]
    args.save_divisor = params["save_divisor"]
    args.eval_divisor = params["eval_divisor"]
    args.toy_eval_divisor = params.get("toy_eval_divisor", 5)
    args.task = params["task"]
    args.epochs = params["epochs"]
    args.lr = params["lr"]
    args.weight_decay = params["weight_decay"]
    args.warm_up_ratio = params["warm_up_ratio"]
    args.scheduler = params["scheduler"] # cosine or linear
    args.optimizer = params["optimizer"] # AdamW or Adam
    args.precision = params["precision"] # bf16 or fp16
    args.model_name = params["model_name"]
    args.n_shot = params["n_shot"]
    args.self_consistency = params["self_consistency"]
    args.max_length = params["max_length"]
    args.lora = params.get("lora", False)

    args.min_eval_epoch = params["min_eval_epoch"]
    args.max_eval_epoch = params["max_eval_epoch"]

    if args.method != "vanilla": # Load method specific args
        with open(f'configs_method/{args.method}.json', 'r') as method_json_file:
            method_json = json.load(method_json_file)
        for key, value in method_json.items():
            setattr(args, key, value)

    exp_name_base = params["exp_name"]
    args.exp_name = f"{exp_name_base}_{args.task}_{args.method}_{args.augmentation_type}_{args.seed}"

    if os.path.exists(args.exp_name):
        while True:
            args.exp_name = args.exp_name + "_"
            if os.path.exists(args.exp_name) == False: break

    os.makedirs(args.exp_name, exist_ok=True)
    os.makedirs(f"{args.exp_name}/{args.log_save}", exist_ok=True)
    os.makedirs(f"{args.exp_name}/{args.ckpt_save}", exist_ok=True)

    configfile = f"{args.exp_name}/{args.log_save}/config.json"

    # Dump the dictionary to a JSON file
    args_dict = vars(args)
    with open(configfile, "w") as json_file:
        json.dump(args_dict, json_file, indent=4)

    train_cmd = f"python train_eval.py --config={configfile} --seed={args.seed} --call_lora=True"
    eval_cmd = f"python evaluate_lora.py --folder_path={args.exp_name}/"

    os.system(train_cmd)
    os.system(eval_cmd)