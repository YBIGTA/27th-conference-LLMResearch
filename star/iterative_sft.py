import argparse
import json
import math
import os
import random
import re
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, get_scheduler

from data.io import append_jsonl
from data.schema import ContrastRationale, Delta, EngineConfig, EvalLine, Rationale, Sample
from engine.stockfish_wrapper import StockfishEngine
from rationale.verify import parse_rationale_text, verify_rationale
from utils import (
    load_jsonl, 
    log_args, 
    collate_fn,
    get_model_tokenizer,
    get_loaded_model_tokenizer,
    setup,
    cleanup,
)

ROOT_DIR = os.path.dirname(__file__)
DEFAULT_MATE_TRAIN = os.path.join(ROOT_DIR, "datasets", "data_mate", "train_stripped.jsonl")
DEFAULT_MATE_TEST = os.path.join(ROOT_DIR, "datasets", "data_mate", "test_stripped.jsonl")
DEFAULT_FENS = []


def _extract_fen(question: str) -> Optional[str]:
    match = re.search(r"board is \"([^\"]+)\"", question)
    return match.group(1) if match else None


def load_mate_fens(path: str) -> List[str]:
    """Load FEN strings from a JSONL file using utils.load_jsonl."""
    fens: List[str] = []
    if not path or not os.path.exists(path):
        return fens
    
    # Use load_jsonl from utils.py
    records = load_jsonl(path)
    for record in records:
        question = record.get("question", "")
        fen = _extract_fen(question)
        if fen:
            fens.append(fen)
    return fens


class SFTDataset(Dataset):
    def __init__(self, tokenizer, samples: List[Sample], include_rationale: bool = False):
        self.tokenizer = tokenizer
        self.samples = samples
        self.include_rationale = include_rationale

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]
        prompt = build_prompt(sample.fen, self.include_rationale)
        target = build_target_text(sample, self.include_rationale)
        enc_prompt = self.tokenizer(prompt, return_tensors="pt")
        enc_target = self.tokenizer(target, return_tensors="pt")
        merged = self.tokenizer(
            prompt + target,
            return_tensors="pt",
        )
        input_ids = merged["input_ids"].squeeze(0)
        attention_mask = merged["attention_mask"].squeeze(0)
        labels = merged["input_ids"].squeeze(0)
        prompt_len = enc_prompt["input_ids"].shape[1]
        labels[:prompt_len] = -100
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


def build_prompt(fen: str, include_rationale: bool) -> str:
    if include_rationale:
        return (
            "You are a chess reasoning agent.\n"
            f"Given the position in FEN: {fen}\n"
            "Propose one move in UCI and a verifiable rationale.\n"
            "Constraints:\n- Provide a short evidence line in UCI (4 to 8 ply).\n"
            "- Provide a contrast move and a refutation line.\n"
            "Output format:\nMOVE: ...\nCLAIM: ...\nEVIDENCE_PV: ...\nCONTRAST_MOVE: ...\nCONTRAST_PV: ...\n"
        )
    return (
        "You are a chess engine assistant.\n"
        f"Given the position in FEN: {fen}\n"
        "Output a single best move in UCI format.\nOutput format:\nMOVE: <uci>\n"
    )


def build_target_text(sample: Sample, include_rationale: bool) -> str:
    if include_rationale and sample.rationale:
        contrast_move = sample.rationale.contrast.alt_move if sample.rationale.contrast else ""
        contrast_pv = " ".join(sample.rationale.contrast.refutation_pv or []) if sample.rationale.contrast else ""
        evidence = " ".join(sample.rationale.evidence_pv or []) if sample.rationale.evidence_pv else ""
        claim = sample.rationale.claim or ""
        return (
            f"MOVE: {sample.chosen}\n"
            f"CLAIM: {claim}\n"
            f"EVIDENCE_PV: {evidence}\n"
            f"CONTRAST_MOVE: {contrast_move}\n"
            f"CONTRAST_PV: {contrast_pv}\n"
        )
    return f"MOVE: {sample.chosen}\n"


def evaluate_model(
    model,
    tokenizer,
    fens: List[str],
    engine: StockfishEngine,
    engine_cfg: EngineConfig,
    include_rationale: bool,
) -> Dict[str, Any]:
    """Compute simple accuracy and cp gap against the engine best moves."""

    if not fens:
        return {"count": 0, "accuracy": 0.0, "avg_cp_gap": None}

    model.eval()
    correct = 0
    total = 0
    cp_gaps: List[int] = []
    for fen in fens:
        prompt = build_prompt(fen, include_rationale)
        encoded = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            output = model.generate(
                **encoded,
                max_new_tokens=64 if include_rationale else 8,
                do_sample=False,
                num_return_sequences=1,
                pad_token_id=tokenizer.eos_token_id,
            )
        text = tokenizer.decode(output[0], skip_special_tokens=True)
        parsed = parse_rationale_text(text)
        move = parsed.get("move")
        if not move:
            continue

        evals = engine.analyze(
            fen,
            multipv=engine_cfg.multipv,
            depth=engine_cfg.depth,
            time_ms=engine_cfg.time_ms,
        )
        if not evals:
            continue

        best_move = evals[0].get("move")
        model_eval = next((ev for ev in evals if ev.get("move") == move), None)
        best_cp = evals[0].get("cp")
        model_cp = model_eval.get("cp") if model_eval else None
        if best_cp is not None and model_cp is not None:
            cp_gaps.append(best_cp - model_cp)

        correct += int(move == best_move)
        total += 1

    accuracy = correct / total if total else 0.0
    avg_cp_gap = sum(cp_gaps) / len(cp_gaps) if cp_gaps else None
    return {"count": total, "accuracy": accuracy, "avg_cp_gap": avg_cp_gap}


def generate_candidates(
    model,
    tokenizer,
    fen: str,
    num_candidates: int,
    include_rationale: bool,
) -> List[Dict[str, Any]]:
    prompt = build_prompt(fen, include_rationale)

    encoded = tokenizer(prompt, return_tensors="pt").to(model.device)

    outputs = model.generate(
        **encoded,
        max_new_tokens=64 if include_rationale else 8,
        do_sample=True,
        top_k=20,
        num_return_sequences=num_candidates,
        pad_token_id=tokenizer.eos_token_id,
    )
    candidates: List[Dict[str, Any]] = []
    for seq in outputs:
        text = tokenizer.decode(seq, skip_special_tokens=True)
        parsed = parse_rationale_text(text)
        move = parsed.get("move")
        rationale = parsed
        candidates.append({"move": move, "rationale": rationale, "raw": text})
    return candidates



def pick_best_eval(evals: List[EvalLine]) -> Optional[EvalLine]:
    if not evals:
        return None
    def score_key(ev: EvalLine) -> float:
        if ev.mate is not None:
            return 100000 if ev.mate > 0 else -100000
        return ev.cp if ev.cp is not None else -math.inf
    return max(evals, key=score_key)


def build_sample(
    fen: str,
    candidates: List[Dict[str, Any]],
    evals: List[Dict[str, Any]],
    engine_cfg: EngineConfig,
) -> Sample:
    eval_lines = [EvalLine(**ev) for ev in evals if ev.get("move")]
    chosen_eval = pick_best_eval(eval_lines)
    model_move = candidates[0]["move"] if candidates else None
    chosen_move = chosen_eval.move if chosen_eval else model_move
    rejected_move = model_move if model_move != chosen_move else None

    model_eval = next((ev for ev in eval_lines if ev.move == model_move), None)
    delta_cp = None
    if chosen_eval and model_eval and chosen_eval.cp is not None and model_eval.cp is not None:
        delta_cp = chosen_eval.cp - model_eval.cp
    delta = Delta(cp=delta_cp, wdl_win_prob=None)

    rationale = None
    parsed = candidates[0].get("rationale") if candidates else None
    if parsed:
        verified, error_type = verify_rationale(fen, parsed)
        contrast = ContrastRationale(
            alt_move=parsed.get("contrast_move"),
            refutation_pv=(parsed.get("contrast_pv") or "").split(),
            note=None,
        )
        rationale = Rationale(
            claim=parsed.get("claim"),
            evidence_pv=(parsed.get("evidence_pv") or "").split(),
            contrast=contrast,
            verified=verified,
            error_type=error_type,
        )

    return Sample(
        id=os.urandom(4).hex(),
        fen=fen,
        side_to_move=fen.split()[1],
        candidates=[cand.get("move") for cand in candidates if cand.get("move")],
        engine=engine_cfg,
        evals=eval_lines,
        chosen=chosen_move,
        rejected=rejected_move,
        delta=delta,
        rationale=rationale,
    )


def train_sft(
    model, 
    tokenizer, 
    samples: List[Sample], 
    include_rationale: bool, 
    lr: float, 
    epochs: int = 1,
    optimizer_type: str = "AdamW",
    weight_decay: float = 0.01,
    warm_up_steps: int = 0,
) -> None:
    """Train model with SFT, using optimizer/scheduler patterns from utils.py."""
    dataset = SFTDataset(tokenizer, samples, include_rationale)
    loader = DataLoader(dataset, batch_size=1, shuffle=True, collate_fn=collate_fn)
    
    # Optimizer setup (similar to get_optimizer_scheduler_step_based in utils.py)
    if optimizer_type == "AdamW":
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_type == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    # Scheduler with warm-up (from utils.py pattern)
    total_steps = len(loader) * epochs
    def lr_lambda(current_step: int):
        if current_step < warm_up_steps:
            return float(current_step) / float(max(1, warm_up_steps))
        return 1.0
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    model.train()
    for _ in range(epochs):
        for batch in loader:
            batch = {k: v.to(model.device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()


def load_model_and_tokenizer(args, rank: int = 0, distributed: bool = False):
    """
    Load model and tokenizer using utils.py functions.
    
    Args:
        args: Training arguments (must have model_name, base_model_path, precision, lora, device)
        rank: GPU rank (0 for single GPU mode)
        distributed: If True, use DDP/FSDP wrapping
    """
    model_path = getattr(args, 'base_model_path', None)
    if model_path and model_path != "null":
        # Use get_loaded_model_tokenizer from utils.py
        model, tokenizer = get_loaded_model_tokenizer(
            args, 
            model_path, 
            args.model_name, 
            rank, 
            eval=False,
            distributed=distributed,
        )
        if rank == 0:
            print(f"[SFT] Model loaded from checkpoint: {model_path}")
    else:
        # Use get_model_tokenizer from utils.py (loads from HuggingFace)
        model, tokenizer = get_model_tokenizer(
            args, 
            args.model_name, 
            rank, 
            eval=False,
            distributed=distributed,
        )
        if rank == 0:
            print(f"[SFT] Model loaded from HuggingFace: {args.model_name}")
    
    # Move to device if not distributed (distributed handles this internally)
    if not distributed:
        device = getattr(args, 'device', 'cpu')
        model.to(device)
    
    return model, tokenizer


def iterative_loop(args: argparse.Namespace) -> None:
    """Main training loop (called via iterative_cli.py or directly)."""
    distributed = getattr(args, 'distributed', False) and getattr(args, 'world_size', 1) > 1
    rank = getattr(args, '_rank', 0)  # Set by fsdp_main for distributed
    
    # Load model and tokenizer using utils.py (get_loaded_model_tokenizer or get_model_tokenizer)
    model, tokenizer = load_model_and_tokenizer(args, rank=rank, distributed=distributed)

    fens = args.fens or load_mate_fens(args.mate_train_path) or DEFAULT_FENS
    engine = StockfishEngine(
        engine_path=args.engine_path,
        default_depth=args.depth,
        default_multipv=args.multipv,
        default_time_ms=args.time_ms,
    )
    engine_cfg = EngineConfig(name="stockfish", depth=args.depth, multipv=args.multipv, time_ms=args.time_ms)

    os.makedirs(args.out_dir, exist_ok=True)
    for iteration in range(args.iterations):
        iteration_samples: List[Sample] = []
        for fen in fens:
            candidates = generate_candidates(
                model,
                tokenizer,
                fen,
                num_candidates=args.num_candidates,
                include_rationale=args.include_rationale,
            )
            candidate_moves = [c["move"] for c in candidates if c.get("move")]
            evals = engine.analyze(fen, moves=candidate_moves, multipv=args.multipv, depth=args.depth, time_ms=args.time_ms)
            sample = build_sample(fen, candidates, evals, engine_cfg)
            iteration_samples.append(sample)
        append_jsonl(os.path.join(args.out_dir, f"iter_{iteration}.jsonl"), [s.to_dict() for s in iteration_samples])
        train_sft(
            model, tokenizer, iteration_samples, args.include_rationale, 
            lr=args.lr, 
            epochs=getattr(args, 'epochs', 1),
            optimizer_type=getattr(args, 'optimizer', 'AdamW'),
            weight_decay=getattr(args, 'weight_decay', 0.01),
            warm_up_steps=getattr(args, 'warm_up_steps', 0),
        )

        train_metrics = evaluate_model(
            model,
            tokenizer,
            fens,
            engine,
            engine_cfg,
            args.include_rationale,
        )
        eval_fens = args.eval_fens or load_mate_fens(args.mate_test_path)
        test_metrics = (
            evaluate_model(
                model,
                tokenizer,
                eval_fens,
                engine,
                engine_cfg,
                args.include_rationale,
            )
            if eval_fens
            else None
        )
        metrics = {"iteration": iteration, "train": train_metrics, "test": test_metrics}
        
        # Use log_args from utils.py for cumulative logging
        metrics_log_path = os.path.join(args.out_dir, "metrics_log.json")
        log_args(metrics_log_path, **metrics)
        
        # Also save individual iteration metrics
        metrics_path = os.path.join(args.out_dir, f"metrics_iter_{iteration}.json")
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)
        print(f"[Iteration {iteration}] train metrics: {train_metrics}")
        if test_metrics:
            print(f"[Iteration {iteration}] test metrics: {test_metrics}")

        model.save_pretrained(os.path.join(args.out_dir, f"model_iter_{iteration}"))
        tokenizer.save_pretrained(os.path.join(args.out_dir, f"model_iter_{iteration}"))

    engine.close()


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Iterative SFT for verifier-only chess reasoning")
    parser.add_argument("--config", type=str, default=None, help="Path to JSON config with hyperparameters")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-3B")
    parser.add_argument("--engine_path", type=str, default="stockfish")
    parser.add_argument("--depth", type=int, default=10)
    parser.add_argument("--multipv", type=int, default=4)
    parser.add_argument("--time_ms", type=int, default=0)
    parser.add_argument("--num_candidates", type=int, default=2)
    parser.add_argument("--iterations", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=1, help="Training epochs per iteration")
    parser.add_argument("--include_rationale", action="store_true")
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--optimizer", type=str, default="AdamW", help="Optimizer type: AdamW or Adam (from utils.py pattern)")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay for optimizer")
    parser.add_argument("--warm_up_steps", type=int, default=0, help="Number of warmup steps for scheduler")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--precision", type=str, default="bf16", help="Model precision: bf16 or fp16")
    parser.add_argument("--distributed", action="store_true", help="Enable distributed training with DDP/FSDP (uses utils.py setup/cleanup)")
    parser.add_argument("--world_size", type=int, default=1, help="Number of GPUs for distributed training")
    parser.add_argument("--lora", type=json.loads, default=None, help="LoRA config as JSON string, e.g. '{\"lora_rank\": 32, \"lora_alpha\": 64, \"lora_dropout\": 0.1}'")
    parser.add_argument("--out_dir", type=str, default="./data/output")
    parser.add_argument("--base_model_path", type=str, default=None, help="Path to base model checkpoint (file or directory). If None, loads from model_name.")
    parser.add_argument("--eval_fens", nargs="*", help="Optional held-out FEN strings for evaluation")
    parser.add_argument("--fens", nargs="*", help="Optional list of FEN strings")
    parser.add_argument(
        "--mate_train_path",
        type=str,
        default=DEFAULT_MATE_TRAIN,
        help="Path to data_mate training split for default FEN sampling",
    )
    parser.add_argument(
        "--mate_test_path",
        type=str,
        default=DEFAULT_MATE_TEST,
        help="Path to data_mate test split for default evaluation FENs",
    )
    known_args, _ = parser.parse_known_args(argv)
    if known_args.config:
        with open(known_args.config, "r", encoding="utf-8") as f:
            config = json.load(f)
        parser.set_defaults(**config)
    return parser.parse_args(argv)


def fsdp_main(rank: int, world_size: int, args: argparse.Namespace) -> None:
    """Entry point for distributed training (called by mp.spawn)."""
    setup(rank, world_size)
    torch.cuda.set_device(rank)
    
    # Set distributed flag
    args.distributed = True
    args.world_size = world_size
    args._rank = rank  # Internal use for distributed
    
    try:
        iterative_loop(args)
    finally:
        cleanup()


def main() -> None:
    """Direct execution entry point (supports --distributed flag)."""
    args = parse_args()
    
    distributed = getattr(args, 'distributed', False) and getattr(args, 'world_size', 1) > 1
    
    if distributed:
        # Distributed training using utils.py setup/cleanup
        print(f"[SFT] Starting distributed training with {args.world_size} GPUs")
        mp.spawn(
            fsdp_main,
            args=(args.world_size, args),
            nprocs=args.world_size,
            join=True,
        )
    else:
        # Single GPU training (same as iterative_cli.py call)
        iterative_loop(args)


if __name__ == "__main__":
    main()
