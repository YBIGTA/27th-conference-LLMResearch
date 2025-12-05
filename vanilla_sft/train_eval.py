import os
import argparse
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import json
from tqdm import tqdm
import gc
from types import SimpleNamespace
import time

from utils.utils import get_model_tokenizer, get_dataloader, get_optimizer_scheduler, setup, cleanup, save_model, save_lora_model, merge_flops_logs
from utils.utils_test import create_labels, test, log_args

def train(args, model, rank, world_size, train_loader, test_loader, tokenizer, optimizer, scheduler, epoch, sampler=None):
    model.train()
    model_forward = model.custom_forward if hasattr(model, 'custom_forward') else model.forward # custom forward if exists

    ddp_loss = torch.zeros(2).to(rank)
    if sampler:
        sampler.set_epoch(epoch)

    # Wrap the dataloader in tqdm for progress tracking
    progress_bar = tqdm(
        enumerate(train_loader),
        total=len(train_loader),
        desc=f"Pretrain | Epoch {epoch} [Rank {rank}]" if args.pretrain else f"Epoch {epoch} [Rank {rank}]",
        position=rank,
        leave=False,
        disable=(rank != 0),  # Only show progress bar on rank 0
    )

    log_interval = max(1, round(len(train_loader) / args.log_divisor))
    if args.save_divisor != 0:
        save_interval = max(1, round(len(train_loader) / args.save_divisor))
    # eval_interval = max(1, round(len(train_loader) / args.eval_divisor))
    log_epoch_interval = 1 / args.eval_divisor
    log_epoch = 0
    log_itr = 1
    eval_points = [int(round(len(train_loader) * i / args.eval_divisor)) for i in range(1, args.eval_divisor)]
    eval_points.append(len(train_loader)) # final eval point
    toy_eval_interval = max(1, round(len(train_loader) / args.toy_eval_divisor))

    for batch_idx, data in progress_bar:
        input_ids = data["input_ids"]
        attention_mask = data["attention_mask"]
        labels = create_labels(input_ids, rank, tokenizer)
        
        # if rank == 0:
        #     input_attention = tokenizer.decode(input_ids[0][attention_mask[0] == 1], skip_special_token=True)
        #     input_labels = tokenizer.decode(labels[0][labels[0] != -100], skip_special_token=True)
        #             # For debug output
        #     target_save = f"{args.exp_name}/{args.log_save}/debug_input.txt"
        #     with open(target_save, 'a+') as new_train_f:
        #         print(f"input_attention: {input_attention} \ninput_labels: {input_labels}", file=new_train_f, end="\n\n")

        input_ids, attention_mask, labels = input_ids.to(rank), attention_mask.to(rank), labels.to(rank)   

        output = model_forward(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = output.loss / args.grad_accum  # Scale loss by accumulation steps
        loss.backward()

        flops_log_file = f"{args.exp_name}/{args.log_save}/flops_train_log_{rank}.json"
        log_args(flops_log_file, epoch=epoch, idx=batch_idx, split="train", batch=args.batch_size, grad_accum=args.grad_accum, context_len=args.batch_size*input_ids.size(1))
          
        if (batch_idx + 1) % args.grad_accum == 0 or (batch_idx + 1) == len(train_loader):
            optimizer.step()
            optimizer.zero_grad()  # Reset gradients
            
        scheduler.step()  # Update learning rate
        # Track loss and sample count
        ddp_loss[0] += loss.item() * args.grad_accum  # Unscaled loss
        ddp_loss[1] += len(input_ids)  # Track total samples

        # saving model
        if args.save_divisor != 0:
            if (batch_idx+1) % save_interval == 0 or (batch_idx + 1) == len(train_loader):
                if args.lora:
                    save_lora_model(args, epoch, batch_idx, model, rank)
                else:
                    save_model(args, model, rank)
        elif args.save_divisor == 0:
            if epoch == args.epochs and (batch_idx + 1) == len(train_loader):
                if args.lora:
                    save_lora_model(args, epoch, batch_idx, model, rank)
                else:
                    save_model(args, model, rank)

        # logging loss
        if (batch_idx+1) % log_interval == 0 or (batch_idx + 1) == len(train_loader):
            dist.all_reduce(ddp_loss, op=dist.ReduceOp.SUM)
            if rank == 0:
                avg_loss = (ddp_loss[0] / ddp_loss[1]).item()
                progress_bar.set_postfix({"Loss": avg_loss})
                trainfile = f"{args.exp_name}/{args.log_save}/train_log.json"
                log_args(trainfile, epoch=epoch, step=batch_idx, loss=avg_loss)

            dist.barrier()
            ddp_loss = torch.zeros(2).to(rank)  # Reset loss tracking
        
        if args.pretrain:   # If it's pretraining, skip evaluation
            continue
        # evaluation
        if epoch == args.min_eval_epoch and (batch_idx + 1) == eval_points[-1]:
            log_epoch = args.min_eval_epoch

            optimizer.step()
            optimizer.zero_grad()
            torch.cuda.empty_cache()
            gc.collect()

            if args.lora: # For very big models, let's save the lora parameters and evaluate later
                save_lora_model(args, epoch, batch_idx, model, rank)
            else:
                test(args, model, rank, world_size, train_loader, test_loader, tokenizer, args.gen_length, args.eval_save, epoch, log_epoch)
                dist.barrier()
                model.train()  # Switch back to training mode
        elif epoch > args.min_eval_epoch and (batch_idx + 1) in eval_points:
            log_epoch = (log_epoch_interval*log_itr) + (epoch-1)
            log_itr += 1

            optimizer.step()
            optimizer.zero_grad()
            torch.cuda.empty_cache()
            gc.collect()

            if args.lora: # For very big models, let's save the lora parameters and evaluate later
                save_lora_model(args, epoch, batch_idx, model, rank)
            else:
                test(args, model, rank, world_size, train_loader, test_loader, tokenizer, args.gen_length, args.eval_save, epoch, log_epoch)
                dist.barrier()
                model.train()  # Switch back to training mode


def fsdp_main(rank, world_size, args):
    setup(rank, world_size)
    torch.cuda.set_device(rank)  # Set the device for this rank

    args.evaluate = False
    
    if args.method == "vanilla":
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

    optimizer, scheduler = get_optimizer_scheduler(args, model, train_loader)
    init_start_event.record()
    for epoch in range(1, args.max_eval_epoch + 1):
        start_time = time.time()
        train(args, model, rank, world_size, train_loader, test_loader, tokenizer, optimizer, scheduler, epoch, sampler=sampler_train)
        elapsed_time = time.time() - start_time
        if rank == 0:
            time_log_file = f"{args.exp_name}/{args.log_save}/elapsed_time_log.json"
            log_args(time_log_file, epoch=epoch, epoch_time=elapsed_time)

    init_end_event.record()

    if rank == 0:
        merge_flops_logs(args, split="train")
        merge_flops_logs(args, split="test")
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
    parser.add_argument('--method', type=str, default="vanilla", help='train method default: vanilla')
    parser.add_argument('--call_lora', type=bool, default=False, help='if lora, just use the config since initialized is done.')
    parser.add_argument('--use_auxiliary', type=bool, default=False, help='if True, pretrain with auxiliary dataset')
    parser.add_argument('--pretrain', type=bool, default=False, help='signal for pretrain')

    args = parser.parse_args()
    torch.manual_seed(args.seed)
    
    params = json.load(open(args.config))

    if args.call_lora: # if called from train_eval_lora, initialized is done just use the loaded config
        args = SimpleNamespace(**params)
        
    else:
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
        args.min_eval_epoch = params["min_eval_epoch"]
        args.max_eval_epoch = params["max_eval_epoch"]
        args.lr = params["lr"]
        args.weight_decay = params["weight_decay"]
        args.warm_up_ratio = params["warm_up_ratio"]
        args.warm_up_steps = params["warm_up_steps"]
        args.scheduler = params["scheduler"] # cosine or linear
        args.optimizer = params["optimizer"] # AdamW or Adam
        args.precision = params["precision"] # bf16 or fp16
        args.model_name = params["model_name"]
        args.n_shot = params["n_shot"]
        args.self_consistency = params["self_consistency"]
        args.max_length = params["max_length"]
        args.lora = params.get("lora", False)
        if args.use_auxiliary:
            args.epochs_pretrain = params["epochs_pretrain"]
        if args.method != "vanilla": # Load method specific args
            with open(f'configs_method/{args.method}.json', 'r') as method_json_file:
                method_json = json.load(method_json_file)
            for key, value in method_json.items():
                setattr(args, key, value)

        exp_name_base = params["exp_name"]
        args.exp_name = f"{exp_name_base}_{args.method}_{args.seed}"

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
