import os
import argparse
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import json
from tqdm import tqdm

from utils.utils import get_model_tokenizer, get_dataloader, get_test_loader_gsmsymb, get_optimizer_scheduler, setup, cleanup, log_args, save_model
from utils.utils_test import create_labels, test

def evaluation(args, model, rank, world_size, train_loader, test_loader, tokenizer):
    epoch, batch_idx = 0, 0
    test(args, model, rank, world_size, train_loader, test_loader, tokenizer, args.gen_length, args.eval_save, epoch, batch_idx)
    dist.barrier()
    if args.task == "gsmsymb":
        # GSM-P1
        test(args, model, rank, world_size, train_loader, args.test_loader_p1, tokenizer, args.gen_length, args.eval_save, epoch, batch_idx)
        dist.barrier()
        # GSM-P2
        test(args, model, rank, world_size, train_loader, args.test_loader_p2, tokenizer, args.gen_length, args.eval_save, epoch, batch_idx)
        dist.barrier()
    if args.task == "gsm8k_to_ASDIV_SVAMP ":
        # SVAMP
        test(args, model, rank, world_size, train_loader, args.test_loader_p1, tokenizer, args.gen_length,
                args.eval_save, epoch, batch_idx)
        dist.barrier()

def fsdp_main(rank, world_size, args):
    setup(rank, world_size)
    torch.cuda.set_device(rank)  # Set the device for this rank

    if args.method == "vanilla" or args.method == "decPr":
        model, tokenizer = get_model_tokenizer(args, args.model_name, rank)
    elif args.method == "neftune":
        model, tokenizer = get_model_tokenizer_neftune(args, args.model_name, rank)
    elif args.method == "hype":
        model, tokenizer = get_model_tokenizer_hype(args, args.model_name, rank)
    elif args.method == "mixout":
        model, tokenizer = get_model_tokenizer_mixout(args, args.model_name, rank)
    else:
        raise ValueError(f"Unknown method: {args.method}")
    
    train_loader, sampler_train, test_loader, sampler_test, train_loader_aug = get_dataloader(args, tokenizer, rank, world_size)
    if args.task == "gsmsymb":
        test_loader_p1, test_loader_p2 = get_test_loader_gsmsymb(args, tokenizer, rank, world_size)
        args.test_loader_p1 = test_loader_p1
        args.test_loader_p2 = test_loader_p2
    if args.task == "gsm8k_to_ASDIV_SVAMP":
        test_loader_p1 = get_test_loader_gsm8k_to_ASDIV_SVAMP(args, tokenizer, rank, world_size)
        args.test_loader_p1 = test_loader_p1

    if args.n_shot > 0:
        prompt_file_path = f"./n_shot_prompts/{args.task}.json"
        with open(prompt_file_path, "r") as f:
            data = json.load(f)
        n_shot_prompts = [item["prompt"] for item in data["n_shot_prompts"]]
        selected_prompts = n_shot_prompts[:args.n_shot]
        n_shot_input = "\n".join(selected_prompts)
        tokenized_prompt = tokenizer(n_shot_input, return_tensors="pt")
        prompt_tokenized_len = tokenized_prompt["input_ids"].shape[1]
        args.n_shot_input = n_shot_input
        args.prompt_tokenized_len = prompt_tokenized_len

    init_start_event = torch.cuda.Event(enable_timing=True)
    init_end_event = torch.cuda.Event(enable_timing=True)

    optimizer, scheduler = get_optimizer_scheduler(args, model, train_loader)
    init_start_event.record()

    evaluation(args, model, rank, world_size, train_loader, test_loader, tokenizer)

    init_end_event.record()

    if rank == 0:
        print(f"CUDA event elapsed time: {init_start_event.elapsed_time(init_end_event) / 1000}sec")
        print(f"{model}")
    cleanup()


if __name__ == '__main__':
    # Training settings
    parser = argparse.ArgumentParser(description='FSDP Finetune')
    parser.add_argument("--config", type=str, required=True, help="Config file location")
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, required=True,help='random seed')
    parser.add_argument('--noise_method', type=bool, default=False, help='train noise method')
    parser.add_argument('--method', type=str, default="vanilla", help='train method default: vanilla')

    # Data augmentation setting
    augmentation_type_list = ['none', 'hard_eda', 'aeda', 'soft_eda', 'adverb_aug', 'adverb_aug_curriculum']
    parser.add_argument('--augmentation_type', type=str, choices=augmentation_type_list, default='none', 
                        help='Whether to use augmented data; Default is none')
    parser.add_argument('--augmentation_eda_alpha_sr', type=float, default=0.1,
                                 help='EDA alpha_sr; Default is 0.1')
    parser.add_argument('--augmentation_eda_alpha_ri', type=float, default=0.1,
                                help='EDA alpha_ri; Default is 0.1')
    parser.add_argument('--augmentation_eda_alpha_rs', type=float, default=0.1,
                                help='EDA alpha_rs; Default is 0.1')
    parser.add_argument('--augmentation_eda_p_rd', type=float, default=0.1,
                                help='EDA p_rd; Default is 0.1')
    parser.add_argument('--augmentation_aeda_alpha', type=float, default=0.1,
                                help='AEDA alpha; Default is 0.1')
    parser.add_argument('--augmentation_label_smoothing', type=float, default=0.2,
                                help='Label smoothing value for SoftEDA; Default is 0.2')

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

    if args.noise_method == True:
        args.noise_path = params["noise_path"]
        args.add_noise = params["add_noise"]

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

    WORLD_SIZE = torch.cuda.device_count()
    mp.spawn(fsdp_main,
        args=(WORLD_SIZE, args),
        nprocs=WORLD_SIZE,
        join=True)
