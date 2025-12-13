import argparse
import logging
import re
from pathlib import Path
from typing import Dict, List

import torch
import yaml
from datasets import Dataset, load_dataset
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

from star.data.star_jsonl import StarSample, load_star_jsonl

logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger(__name__)


DEFAULT_INSTRUCTION = (
    "You are an expert chess player. Given the FEN, provide the best move in UCI and explain briefly."
)

FEN_REGEX = re.compile(r"([prnbqkPRNBQK1-8]+\/[prnbqkPRNBQK1-8]+\/[prnbqkPRNBQK1-8]+\/[prnbqkPRNBQK1-8]+\/[prnbqkPRNBQK1-8]+\/[prnbqkPRNBQK1-8]+\/[prnbqkPRNBQK1-8]+\/[prnbqkPRNBQK1-8]+\s+[wb]\s+(?:-|[KQkq]+)\s+(?:-|[a-h][36])\s+[0-9]+\s+[0-9]+)")


def load_yaml(config_path: str) -> Dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def apply_overrides(config: Dict, overrides: List[str]) -> Dict:
    for override in overrides:
        if "=" not in override:
            continue
        key, value = override.split("=", 1)
        ref = config
        parts = key.split(".")
        for part in parts[:-1]:
            ref = ref.setdefault(part, {})
        ref[parts[-1]] = yaml.safe_load(value)
    return config


def build_model_and_tokenizer(model_cfg: Dict):
    quant_config = None
    if model_cfg.get("load_in_4bit"):
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=getattr(torch, model_cfg.get("torch_dtype", "float16")),
        )
    tokenizer = AutoTokenizer.from_pretrained(model_cfg["name_or_path"])
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_cfg["name_or_path"],
        quantization_config=quant_config,
        torch_dtype=getattr(torch, model_cfg.get("torch_dtype", "float16")),
        device_map="auto",
    )
    if model_cfg.get("lora", {}).get("enable"):
        lora_cfg = model_cfg["lora"]
        peft_cfg = LoraConfig(
            r=lora_cfg.get("r", 16),
            lora_alpha=lora_cfg.get("alpha", 32),
            target_modules=["q_proj", "v_proj"],
            lora_dropout=lora_cfg.get("dropout", 0.05),
        )
        model = get_peft_model(model, peft_cfg)
    return model, tokenizer


def mate_dataset(mate_root: str) -> Dataset:
    train_path = Path(mate_root) / "train_stripped.jsonl"
    if not train_path.exists():
        train_path = Path("star") / Path(mate_root) / "train_stripped.jsonl"
    return load_dataset("json", data_files=str(train_path))["train"]


def format_supervised_example(example, add_rationale: bool) -> str:
    fen = example.get("fen") or example.get("question", "")
    match = FEN_REGEX.search(fen)
    fen_value = match.group(1) if match else fen
    answer = example.get("move") or example.get("answer", "")
    rationale = example.get("rationale", "") if add_rationale else ""
    fen_text = fen_value if fen_value else fen
    prompt = f"{DEFAULT_INSTRUCTION}\nFEN: {fen_text}\n"
    if rationale:
        prompt += f"RATIONALE: {rationale}\n"
    prompt += "MOVE:"
    return prompt + f" {answer}"


def build_training_texts(
    mode: str,
    mate_root: str,
    star_jsonl_path: str,
    use_original_mate: bool,
    success_weight: float,
    fallback_weight: float,
) -> List[str]:
    texts: List[str] = []
    if mode == "baseline_sft" or use_original_mate:
        mate_ds = mate_dataset(mate_root)
        for row in mate_ds:
            texts.append(format_supervised_example(row, add_rationale=False))
    if mode == "star_sft":
        star_samples = load_star_jsonl(star_jsonl_path)
        for sample in star_samples:
            weight = success_weight if sample.source == "success" else fallback_weight
            for _ in range(max(1, int(round(weight)))):
                texts.append(format_supervised_example(dataclass_to_dict(sample), add_rationale=True))
    return texts


def dataclass_to_dict(sample: StarSample) -> Dict:
    return {
        "fen": sample.fen,
        "move": sample.move,
        "rationale": sample.rationale,
    }


def tokenize_dataset(tokenizer, texts: List[str], max_length: int) -> Dataset:
    tokenized = tokenizer(
        texts,
        padding=False,
        truncation=True,
        max_length=max_length,
    )
    tokenized["labels"] = tokenized["input_ids"].copy()
    return Dataset.from_dict(tokenized)


def run_training(config: Dict) -> None:
    model, tokenizer = build_model_and_tokenizer(config["model"])
    train_cfg = config.get("train", {})
    texts = build_training_texts(
        mode=train_cfg.get("mode", "baseline_sft"),
        mate_root=config["data"].get("mate_root", "dataset/data_mate"),
        star_jsonl_path=config["data"].get("output_star_jsonl"),
        use_original_mate=train_cfg.get("mix", {}).get("use_original_mate", True),
        success_weight=train_cfg.get("mix", {}).get("star_success_weight", 1.0),
        fallback_weight=train_cfg.get("mix", {}).get("fallback_weight", 0.2),
    )
    if not texts:
        LOGGER.warning("No training texts found. Check data paths and generation outputs.")
        return
    dataset = tokenize_dataset(tokenizer, texts, max_length=train_cfg.get("max_length", 512))

    training_args = TrainingArguments(
        output_dir=train_cfg.get("output_dir", "outputs/star_sft"),
        per_device_train_batch_size=train_cfg.get("batch_size", 4),
        gradient_accumulation_steps=train_cfg.get("grad_accum", 8),
        learning_rate=train_cfg.get("lr", 1e-4),
        max_steps=train_cfg.get("max_steps", 20000),
        logging_steps=10,
        save_steps=train_cfg.get("save_steps", 1000),
        evaluation_strategy="no",
        bf16=train_cfg.get("torch_dtype", "bfloat16") == "bfloat16",
    )

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    trainer = Trainer(model=model, args=training_args, train_dataset=dataset, data_collator=data_collator)
    trainer.train()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train baseline or STaR SFT model")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    args, unknown = parser.parse_known_args()
    args.overrides = unknown
    return args


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config)
    if getattr(args, "overrides", None):
        config = apply_overrides(config, args.overrides)
    run_training(config)


if __name__ == "__main__":
    main()
