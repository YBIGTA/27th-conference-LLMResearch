import os
import json
import torch
import torch.optim as optim
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.nn.parallel import DistributedDataParallel as DDP

from torch.distributed.fsdp import MixedPrecision, FullStateDictConfig, StateDictType, FullStateDictConfig, MixedPrecision
from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy
import functools
from datasets import load_from_disk, load_dataset, concatenate_datasets
from transformers import AutoModelForCausalLM, AutoTokenizer, get_scheduler
import logging
import random
import glob


def fsdp_wrap(args, model, rank):
    if args.precision == "bf16":
        mixed_precision_policy = MixedPrecision(
            param_dtype=torch.bfloat16,  # Model parameters in bfloat16
            reduce_dtype=torch.bfloat16,  # Gradient reduction in bfloat16
            buffer_dtype=torch.bfloat16  # Buffers in bfloat16
            )
    elif args.precision == "fp16":
        mixed_precision_policy = MixedPrecision(
            param_dtype=torch.float16,  # Model parameters in bfloat16
            reduce_dtype=torch.float16,  # Gradient reduction in bfloat16
            buffer_dtype=torch.float16  # Buffers in bfloat16
            )

    cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
    args.cfg = cfg

    my_auto_wrap_policy = functools.partial(
        size_based_auto_wrap_policy, min_num_params=100
    )

    model = FSDP(
        model,
        auto_wrap_policy=my_auto_wrap_policy,
        mixed_precision=mixed_precision_policy,
        device_id=torch.device(rank),  # Specify the device
        )
    return model

def get_model_tokenizer(args, model_name, rank):
    if args.precision == "bf16":
        model_dtype = torch.bfloat16
    if model_name == "Qwen/Qwen2.5-3B" :
        model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True,torch_dtype=torch.bfloat16)
    else :
        model = AutoModelForCausalLM.from_pretrained(model_name,torch_dtype=model_dtype)
    model.to(model_dtype)

    if args.lora:
        from peft import LoraConfig, TaskType, get_peft_model
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM, inference_mode=False,
            r=args.lora["lora_rank"],
            lora_alpha=args.lora["lora_alpha"],
            lora_dropout=args.lora["lora_dropout"],
        )
        model = get_peft_model(model, peft_config)
        model.to(model_dtype)
        if rank == 0:
            if isinstance(model, torch.nn.parallel.DistributedDataParallel):
                model.module.print_trainable_parameters()
            else:
                model.print_trainable_parameters()

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    if args.evaluate: # called from evaluate_lora.py
        dist.barrier()
        model = model.to(rank)
        model = DDP(model, device_ids=[rank], output_device=rank) # Wrap with DDP
        if rank == 0:
            print("Using wrapped model with DDP")
    else:
        model = fsdp_wrap(args, model, rank)
    return model, tokenizer

def get_optimizer_scheduler(args, model, train_loader):
    if args.optimizer == "AdamW":
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == "Adam":
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    num_training_steps = len(train_loader) * args.epochs
    num_warmup_steps = args.warm_up_steps # if int(args.warm_up_ratio * num_training_steps)
    
    scheduler = get_scheduler(name=args.scheduler, optimizer=optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps)
    return optimizer, scheduler

def get_dataloader(args, model, tokenizer, rank, world_size):
    dataset_train, dataset_test = None, None

    preprocess_function = preprocess_function_default

    if (args.task == "gsm8k"):
        dataset_train = load_dataset("json", data_files="../datasets/data_gsm8k/train_cleaned.jsonl")["train"]
        dataset_test = load_dataset("json", data_files="../datasets/data_gsm8k/test.jsonl")["train"]
        
        dataset_train = dataset_train.map(lambda examples: preprocess_function(args, examples, tokenizer, "train"), batched=True)
        dataset_test = dataset_test.map(lambda examples: preprocess_function(args, examples, tokenizer, "test"), batched=True)

    elif args.task == "logiqa":
        dataset_train = load_dataset("json", data_files="../datasets/data_logiqa2_split/train.jsonl")["train"]
        dataset_test = load_dataset("json", data_files="../datasets/data_logiqa2_split/test.jsonl")["train"]

        dataset_train = dataset_train.map(lambda examples: preprocess_function(args, examples, tokenizer, "train"), batched=True)
        dataset_test = dataset_test.map(lambda examples: preprocess_function(args, examples, tokenizer, "test"), batched=True)
    
    elif (args.task == "cladder"):
        dataset_train = load_dataset("json", data_files="../datasets/data_cladder_split/cladder_train_long.jsonl")["train"]
        dataset_test = load_dataset("json", data_files="../datasets/data_cladder_split/cladder_test.jsonl")["train"]
        
        dataset_train = dataset_train.map(lambda examples: preprocess_function(args, examples, tokenizer, "train"), batched=True)
        dataset_test = dataset_test.map(lambda examples: preprocess_function(args, examples, tokenizer, "test"), batched=True)

    elif args.task == "arc_challenge":
        dataset_train = load_dataset("json", data_files="../datasets/data_arc_challenge/train_val_combined.jsonl")["train"]
        dataset_test = load_dataset("json", data_files="../datasets/data_arc_challenge/test_new.jsonl")["train"]
        
        dataset_train = dataset_train.map(lambda examples: preprocess_function(args, examples, tokenizer, "train"), batched=True)
        dataset_test = dataset_test.map(lambda examples: preprocess_function(args, examples, tokenizer, "test"), batched=True)

    elif args.task == "cqa":
        dataset_train = load_dataset("json", data_files="../datasets/CommonsenseQA/train_rand_split.jsonl")["train"]
        dataset_test = load_dataset("json", data_files="../datasets/CommonsenseQA/test.jsonl")["train"]
        
        dataset_train = dataset_train.map(lambda examples: preprocess_function(args, examples, tokenizer, "train"), batched=True)
        dataset_test = dataset_test.map(lambda examples: preprocess_function(args, examples, tokenizer, "test"), batched=True)

    elif args.task == "asdiv":
        dataset_train = load_dataset("json", data_files="../datasets/data_asdiv/train_data.jsonl")["train"]
        dataset_test = load_dataset("json", data_files="../datasets/data_asdiv/test_data.jsonl")["train"]
        
        dataset_train = dataset_train.map(lambda examples: preprocess_function(args, examples, tokenizer, "train"), batched=True)
        dataset_test = dataset_test.map(lambda examples: preprocess_function(args, examples, tokenizer, "test"), batched=True)

    elif args.task == "svamp":
        dataset_train = load_dataset("json", data_files="../datasets/data_svamp/train.jsonl")["train"]
        dataset_test = load_dataset("json", data_files="../datasets/data_svamp/test.jsonl")["train"]
        
        dataset_train = dataset_train.map(lambda examples: preprocess_function(args, examples, tokenizer, "train"), batched=True)
        dataset_test = dataset_test.map(lambda examples: preprocess_function(args, examples, tokenizer, "test"), batched=True)
        
    elif args.task == "aqua_rat":
        dataset_train = load_dataset("json", data_files="../datasets/data_aqua_rat/train_new.jsonl")["train"]
        dataset_test = load_dataset("json", data_files="../datasets/data_aqua_rat/test_new.jsonl")["train"]
        
        # Preprocess datasets
        dataset_train = dataset_train.map(lambda examples: preprocess_function(args, examples, tokenizer, "train"), batched=True)
        dataset_test = dataset_test.map(lambda examples: preprocess_function(args, examples, tokenizer, "test"), batched=True)    

    
    elif args.task == "anli_r1":
        dataset_train = load_dataset("json", data_files="../datasets/data_anli/train_r1_modified.jsonl")["train"]
        dataset_test = load_dataset("json", data_files="../datasets/data_anli/test_r1_modified.jsonl")["train"]

        # Preprocess datasets
        dataset_train = dataset_train.map(lambda examples: preprocess_function(args, examples, tokenizer, "train"), batched=True)
        dataset_test = dataset_test.map(lambda examples: preprocess_function(args, examples, tokenizer, "test"), batched=True)

    elif args.task == "mate":
        dataset_train = load_dataset("json", data_files="../datasets/data_mate/train.jsonl")["train"]
        dataset_test = load_dataset("json", data_files="../datasets/data_mate/test.jsonl")["train"]

        # Preprocess datasets
        dataset_train = dataset_train.map(lambda examples: preprocess_function(args, examples, tokenizer, "train"), batched=True)
        dataset_test = dataset_test.map(lambda examples: preprocess_function(args, examples, tokenizer, "test"), batched=True)

    elif args.task == "numglue":
        dataset_train = load_dataset("json", data_files="../datasets/data_numglue/type_4/converted_train.jsonl")["train"]
        dataset_test = load_dataset("json", data_files="../datasets/data_numglue/type_4/converted_test.jsonl")["train"]
        # Preprocess datasets
        dataset_train = dataset_train.map(lambda examples: preprocess_function(args, examples, tokenizer, "train"), batched=True)
        dataset_test = dataset_test.map(lambda examples: preprocess_function(args, examples, tokenizer, "test"), batched=True)

    elif args.task == "gsm8k_svamp":
        dataset_train = load_dataset("json", data_files="../datasets/data_gsm8k/train.jsonl")["train"]
        dataset_test = load_dataset("json", data_files="../datasets/data_svamp/test.jsonl")["train"]

        dataset_train = dataset_train.map(lambda examples: preprocess_function(args, examples, tokenizer, "train", special_task="gsm8k"), batched=True)
        dataset_test = dataset_test.map(lambda examples: preprocess_function(args, examples, tokenizer, "test", special_task="svamp"), batched=True)

    # # For debugging purposes, limit the number of examples
    # dataset_train = dataset_train.select(range(min(128, len(dataset_train))))
    # dataset_test = dataset_test.select(range(min(128, len(dataset_test))))

    print(f"Loaded dataset for task {args.task}: {len(dataset_train) if dataset_train else 'None'} samples")    # for debugging
    sampler_train = DistributedSampler(dataset_train, rank=rank, num_replicas=world_size, shuffle=True)
    sampler_test = DistributedSampler(dataset_test, rank=rank, num_replicas=world_size)
    train_kwargs = {'batch_size': args.batch_size, 'sampler': sampler_train, 'collate_fn': collate_fn}
    test_kwargs = {'batch_size': args.test_batch_size, 'sampler': sampler_test, 'collate_fn': collate_fn}
    cuda_kwargs = {'num_workers': 4,
                   'pin_memory': True,
                   'shuffle': False}

    train_kwargs.update(cuda_kwargs)
    test_kwargs.update(cuda_kwargs)

    train_loader = torch.utils.data.DataLoader(dataset_train,**train_kwargs)
    test_loader = torch.utils.data.DataLoader(dataset_test, **test_kwargs)
        
    return train_loader, sampler_train, test_loader, sampler_test

def preprocess_function_default(args, examples, tokenizer, split, special_task=None): 
    eos = tokenizer.eos_token
    if args.task == "logiqa":
        # Tokenize the input text
        if split == "train":
            combined_texts = [f"{t}\nQ: {q}\nOptions:\n0.{o[0]}\n1.{o[1]}\n2.{o[2]}\n3.{o[3]}\nA: {a}{eos}" for t, q, o, a in zip(examples["text"], examples["question"], examples["options"], examples["answer"])]
        else:
            combined_texts = [f"{t}\nQ: {q}\nOptions:\n0.{o[0]}\n1.{o[1]}\n2.{o[2]}\n3.{o[3]}\nA: " for t, q, o in zip(examples["text"], examples["question"], examples["options"])]
        
        # Tokenize the combined texts
        tokenized = tokenizer(
            combined_texts,
            padding="max_length",  # Pad to max length
            truncation=True,  # Truncate if exceeding max length
            max_length=args.max_length,  # Adjust max length as needed
        )
        
        # Add the original question and answer to the tokenized data
        tokenized["text"] = examples["text"]
        tokenized["question"] = examples["question"]
        tokenized["options"] = examples["options"]
        tokenized["answer"] = examples["answer"]

    elif args.task == "arc_challenge":  
        combined_texts = []
        
        for q, choices, ans in zip(examples["question"], examples["choices"], examples.get("answerKey", [""] * len(examples["question"]))):
            options_text = "\n".join([f"{label}. {opt}" for label, opt in zip(choices["label"], choices["text"])])
            if split == "train":
                combined_texts.append(f"Q: {q}\nOptions:\n{options_text}\nA: {ans}")
            else:
                combined_texts.append(f"Q: {q}\nOptions:\n{options_text}\nA: ")

        tokenized = tokenizer(
            combined_texts,
            padding="max_length",
            truncation=True,
            max_length=args.max_length
        )

        # Label 생성
        if split == "train":
            if "answerKey" not in examples:
                raise KeyError(f"Error: 'answerKey' field is missing in dataset! Available keys: {examples.keys()}")
            tokenized["labels"] = [ord(ans) - ord("A") for ans in examples["answerKey"]]

        tokenized["question"] = examples["question"]
        tokenized["options"] = [c["text"] for c in examples["choices"]]
        tokenized["answer"] = examples["answerKey"]

    elif args.task == "cqa":  
        combined_texts = []
        # question_components = examples["question"]
        
        for text, ans in zip(examples["question"], examples["answerKey"]):
            q = text['stem']
            choices = text['choices']
            options_text = "\n".join([f'({choice["label"]}) {choice["text"]}' for choice in choices])
            if split == "train":
                combined_texts.append(f"Q: {q}\nOptions:\n{options_text}\nA: {ans}")
            else:
                combined_texts.append(f"Q: {q}\nOptions:\n{options_text}\nA: ")

        tokenized = tokenizer(
            combined_texts,
            padding="max_length",
            truncation=True,
            max_length=args.max_length
        )

        tokenized["question"] = examples["question"]
        tokenized["answer"] = examples["answerKey"]
        
    elif args.task == "aqua_rat":  
        combined_texts = []
        
        for q, options, ans in zip(examples["question"], examples["options"], examples["correct"]):
            options_text = "\n".join([f"{opt}" for opt in  options])
            if split == "train":
                combined_texts.append(f"Q: {q}\nOptions:\n{options_text}\nA: {ans}")
            else:
                combined_texts.append(f"Q: {q}\nOptions:\n{options_text}\nA: ")

        tokenized = tokenizer(
            combined_texts,
            padding="max_length",
            truncation=True,
            max_length=args.max_length
        )

        tokenized["question"] = examples["question"]
        tokenized["options"] = examples["options"]
        tokenized["answer"] = examples["correct"]
        tokenized["rationale"] = examples["rationale"] 

    elif args.task == "asdiv":
        if split == "train":
            # combined_texts = [f"Q: {t} {q}\nA: {f}\n####{a}{eos}" for t, q, f, a in zip(examples["body"], examples["question"], examples["formula"], examples["answer"])]
            combined_texts = [f"Q: {t} {q}\nA: #### {a}{eos}" for t, q, a in zip(examples["body"], examples["question"], examples["answer"])]
        else:
            combined_texts = [f"Q: {t} {q}\nA: " for t, q in zip(examples["body"], examples["question"])]

        tokenized = tokenizer(
            combined_texts,
            padding="max_length",
            truncation=True,
            max_length=args.max_length
        )

        tokenized["text"] = examples["body"]
        tokenized["question"] = examples["question"]
        # tokenized["rationale"] = examples["formula"]
        tokenized["answer"] = examples["answer"]

    elif args.task == "svamp":
        if split == "train":
            combined_texts = [f"Q: {q}\nA: {eq}\n#### {a}{eos}" for q, eq, a in zip(examples["question_concat"], examples["Equation"], examples["answer"])]
        else:
            combined_texts = [f"Q: {q}\nA: " for q in examples["question_concat"]]
        
        tokenized = tokenizer(
            combined_texts,
            padding="max_length",  # Pad to max length
            truncation=True,  # Truncate if exceeding max length
            max_length=args.max_length,  # Adjust max length as needed
        )
        
        tokenized["question"] = examples["question_concat"]
        tokenized["rationale"] = examples["Equation"]
        tokenized["answer"] = examples["answer"]

    elif args.task == "mate":
        if split == "train":
            eos = tokenizer.eos_token or ""
            instruct = examples["instruction"] 
            questions = examples["question"]      # or "input" 등 실제 컬럼 이름에 맞춰 수정
            # methods   = examples["Method"]        # 실제 컬럼 이름 확인
            # tactics   = examples["Tactic"]        # 실제 컬럼 이름 확인
            answers   = examples["answer"]        # 최종 수 (예: "b2b1")

            combined_texts = [
                f"Instruct: {i}\n"
                f"Q: {q}\n"
                # f"A: Method: {m}\n"
                # f"Tactic: {t}\n"
                f"#### {a}{eos}"
                for i, q,  a in zip(instruct,questions, answers)
            ] 
        else:
            combined_texts = [f"Instruct: {i}\n" f"Q: {q}\nA: " for i, q in zip(examples["instruction"] ,examples["question"])]

        
        tokenized = tokenizer(
            combined_texts,
            padding="max_length",  # Pad to max length
            truncation=True,  # Truncate if exceeding max length
            max_length=args.max_length,  # Adjust max length as needed
        )
        tokenized["instruct"] = examples["instruction"]
        tokenized["question"] = examples["question"]
        # tokenized["rationale"] = examples["Method"]
        # tokenized["rationale"] = examples["Tactic"] 
        tokenized["answer"] = examples["answer"]
        
    elif args.task == "anli_r1":
        combined_texts = []
        
        for q,h, choices, ans in zip(examples["premise"], examples["hypothesis"],examples["choices"] ,examples["label"]):
            labels = choices["label"]  # [0, 1, 2]
            texts = choices["text"]    # ["entailment", "neutral", "contradiction"]
            options_text = "\n".join([f'({label}) {text}' for label, text in zip(labels, texts)])
            if split == "train":
                combined_texts.append(f"Q: {q} {h}\nOptions:\n{options_text}\nA: {ans}")
            else:
                combined_texts.append(f"Q: {q} {h}\nOptions:\n{options_text}\nA: ")

        tokenized = tokenizer(
            combined_texts,
            padding="max_length",  # Pad to max length
            truncation=True,  # Truncate if exceeding max length
            max_length=args.max_length,  # Adjust max length as needed
        )
        tokenized["question"] = [p + " " + h for p, h in zip(examples["premise"], examples["hypothesis"])]        
        tokenized["rationale"] = examples["reason"]
        tokenized["answer"] = examples["label"]    

    elif args.task == "gsm8k_svamp":
        if split == "train":
            combined_texts = [f"Q: {q}\nA: {a}" for q, a in zip(examples["question"], examples["answer"])]

            # Tokenize the combined texts
            tokenized = tokenizer(
                combined_texts,
                padding="max_length",  # Pad to max length
                truncation=True,  # Truncate if exceeding max length
                max_length=args.max_length,  # Adjust max length as needed
            )

            # Add the original question and answer to the tokenized data
            tokenized["question"] = examples["question"]
            tokenized["answer"] = examples["answer"]

        else:
            combined_texts = [f"Q: {q}\nA: " for q in examples["question_concat"]]
                
            tokenized = tokenizer(
                combined_texts,
                padding="max_length",  # Pad to max length
                truncation=True,  # Truncate if exceeding max length
                max_length=args.max_length,  # Adjust max length as needed
            )

            tokenized["question"] = examples["question_concat"]
            tokenized["rationale"] = examples["Equation"]
            tokenized["answer"] = examples["answer"]
   
    # gsm8k, cladder
    else:
        # Tokenize the input text and keep "question" and "answer"
        if split == "train":
            combined_texts = [f"Q: {q}\nA: {a}{eos}" for q, a in zip(examples["question"], examples["answer"])]
        else:
            combined_texts = [f"Q: {q}\nA: " for q in examples["question"]]
        
        # Tokenize the combined texts
        tokenized = tokenizer(
            combined_texts,
            padding="max_length",  # Pad to max length
            truncation=True,  # Truncate if exceeding max length
            max_length=args.max_length,  # Adjust max length as needed
        )
        
        # Add the original question and answer to the tokenized data
        tokenized["question"] = examples["question"]
        tokenized["answer"] = examples["answer"]

    log_truncation_warnings(args, combined_texts, tokenizer, log_point="Simple data load")
    
    return tokenized

def align_field_order(dataset, reference_fields):
    def reorder_fields(example):
        return {field: example[field] for field in reference_fields if field in example}
    return dataset.map(reorder_fields)

def collate_fn(batch):
    # Convert input_ids and attention_mask to tensors
    input_ids = torch.tensor([item["input_ids"] for item in batch], dtype=torch.long)
    attention_mask = torch.tensor([item["attention_mask"] for item in batch], dtype=torch.long)
    questions = [item["question"] for item in batch]
    answers = [item["answer"] for item in batch]
    
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "questions": questions,
        "answers": answers,
    }

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    # initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def log_args(output_file, **kwargs):
    # Load existing log data or initialize an empty dictionary
    if os.path.exists(output_file):
        try:
            with open(output_file, "r") as json_file:
                logs = json.load(json_file)
        except (FileNotFoundError, json.JSONDecodeError):
            logs = []
    else:
        logs = []
    
    logs.append(kwargs)
    with open(output_file, "w") as json_file:
        json.dump(logs, json_file, indent=4)

def log_truncation_warnings(args, prompts, tokenizer, log_point):
    # Configure logging (global configuration)
    logging.basicConfig(
        level=logging.WARNING,  # Set the default log level to WARNING
        format="%(levelname)s - %(message)s",  # Define the log message format
        handlers=[
            logging.FileHandler(f"{args.exp_name}/truncation_warnings.log")  # Save logs to a file
        ]
    )
    # Compare the original lengths of each input to the maximum length
    original_lengths = [len(tokenizer(prompt, add_special_tokens=False)["input_ids"]) for prompt in prompts]
    truncated = [original > args.max_length for original in original_lengths]

    if any(truncated):
        truncated_count = sum(truncated)
        logging.warning(f"{truncated_count} input(s) exceeded max_length and were truncated.")
        for idx, (prompt, length, is_truncated) in enumerate(zip(prompts, original_lengths, truncated)):
            if is_truncated:
                logging.warning(f"Log point: {log_point}\n--Truncated Prompt {idx+1} (Token Length {length} > Max Length {args.max_length})--\n{prompt}")

def merge_flops_logs(args, split):
    if split == "train":
        flops_log_files = glob.glob(f"{args.exp_name}/{args.log_save}/flops_train_log_*.json")
    elif split == "test":
        flops_log_files = glob.glob(f"{args.exp_name}/{args.log_save}/flops_test_log_*.json")
    merged_data = []

    for file in flops_log_files:
        with open(file, 'r') as f:
            try:
                data = json.load(f)
                merged_data.extend(data)
            except json.JSONDecodeError:
                continue

    with open(f"{args.exp_name}/{args.log_save}/flops_log.json", 'w') as f:
        json.dump(merged_data, f, indent=2)

def save_model(args, model, rank):
    dist.barrier()
    with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, args.cfg):
        state_dict = model.state_dict()
        if rank==0:
            savefile = f"{args.exp_name}/{args.ckpt_save}/final_model.pth"
            torch.save(state_dict, savefile)
            print("saved model in:", savefile)
    dist.barrier()

def save_lora_model(args, epoch, step, model, rank):
    dist.barrier()
    with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, args.cfg):
        state_dict = model.state_dict()
        if rank == 0:
            # Extract only LoRA parameters
            lora_state_dict = {
                k: v for k, v in state_dict.items() if "lora_" in k
            }

            # Save LoRA parameters to a file
            save_path = f"{args.exp_name}/{args.ckpt_save}/lora-weights_{epoch}_{step}.pth"
            torch.save(lora_state_dict, save_path)
            print(f"LoRA parameters saved in: {save_path}")
    dist.barrier()

def load_lora_state(args, model, lora_path, rank):
    # Load LoRA state dictionary
    lora_state_dict = torch.load(lora_path, map_location="cpu")
    model_state_dict = model.state_dict()
    updated_lora_state_dict = {}

    for key in lora_state_dict.keys():
        new_key = "module." + key if "module." + key in model_state_dict else key
        updated_lora_state_dict[new_key] = lora_state_dict[key]

    # Load LoRA parameters into the model
    model.load_state_dict(updated_lora_state_dict, strict=False)  # Allow missing keys
    dist.barrier()
    
    return model
