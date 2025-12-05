import os
import argparse
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import json

from utils.utils import get_model_tokenizer, get_dataloader, setup, cleanup, load_lora_state
from utils.utils_test import test

def evaluate(args, model, rank, world_size, train_loader, test_loader, tokenizer, epoch, batch_idx):
    test(args, model, rank, world_size, train_loader, test_loader, tokenizer, args.gen_length, args.eval_save, epoch)
    dist.barrier()

def fsdp_main(rank, world_size, args):
    setup(rank, world_size)
    torch.cuda.set_device(rank)  # Set the device for this rank

    args.evaluate = True

    if args.method == "vanilla" or args.method == "decPr":
        model, tokenizer = get_model_tokenizer(args, args.model_name, rank)
    else:
        raise ValueError(f"Unknown method: {args.method}")
    
    train_loader, sampler_train, test_loader, sampler_test = get_dataloader(args, model, tokenizer, rank, world_size)

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

    init_start_event.record()

    chkpoints = []
    ckpt_path = f"{args.exp_name}/{args.ckpt_save}/"
    for root, dirs, files in os.walk(ckpt_path):
        for file in files:
            chkpoints.append(os.path.join(root, file))

    for chkpoint in chkpoints:
        if chkpoint.endswith(".pth"):
            model = load_lora_state(args, model, chkpoint, rank)
        else:
            raise ValueError(f"Unknown checkpoint format: {chkpoint}")
        
        chkpoint_name = chkpoint.split("/")[-1]
        epoch = int(chkpoint_name.split("_")[1])
        batch_idx = int(chkpoint_name.split("_")[2].split(".")[0])

        evaluate(args, model, rank, world_size, train_loader, test_loader, tokenizer, epoch, batch_idx)
        if rank == 0:
            print(f"Checkpoint {chkpoint} evaluated")

    init_end_event.record()

    if rank == 0:
        print(f"CUDA event elapsed time: {init_start_event.elapsed_time(init_end_event) / 1000}sec")
        print(f"{model}")
    cleanup()


if __name__ == '__main__':
    # Training settings
    parser = argparse.ArgumentParser(description='FSDP LoRA Evaluation')
    parser.add_argument("--folder_path", type=str, required=True, help="Folder path to the log / checkpoints files")
    parser.add_argument("--test_bs_manual", type=int, default=0, help="Change test batch size manually")

    args = parser.parse_args()
    test_bs_manual = args.test_bs_manual
    params = json.load(open(args.folder_path + "/log/config.json"))
    args = argparse.Namespace(**params)

    if test_bs_manual > 0:
        args.test_batch_size = test_bs_manual

    torch.manual_seed(args.seed)
    WORLD_SIZE = torch.cuda.device_count()
    mp.spawn(fsdp_main,
        args=(WORLD_SIZE, args),
        nprocs=WORLD_SIZE,
        join=True)
