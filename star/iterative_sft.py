import argparse
import math
import os
import random
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, get_scheduler

from data.io import append_jsonl
from data.schema import ContrastRationale, Delta, EngineConfig, EvalLine, Rationale, Sample
from engine.stockfish_wrapper import StockfishEngine
from rationale.verify import parse_rationale_text, verify_rationale

DEFAULT_FENS = [
    "r1bqkbnr/pppppppp/n7/8/3P4/5N2/PPP1PPPP/RNBQKB1R b KQkq - 1 2",
]


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


def generate_candidates(
    model,
    tokenizer,
    fen: str,
    num_candidates: int,
    include_rationale: bool,
) -> List[Dict[str, Any]]:
    prompt = build_prompt(fen, include_rationale)
    encoded = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(
        **{k: v for k, v in encoded.items()},
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


def train_sft(model, tokenizer, samples: List[Sample], include_rationale: bool, lr: float, epochs: int = 1) -> None:
    dataset = SFTDataset(tokenizer, samples, include_rationale)
    loader = DataLoader(dataset, batch_size=1, shuffle=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=len(loader) * epochs)
    model.train()
    for _ in range(epochs):
        for batch in loader:
            batch = {k: v.to(model.device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()


def iterative_loop(args: argparse.Namespace) -> None:
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model_name)
    model.to(args.device)

    fens = args.fens or DEFAULT_FENS
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
        train_sft(model, tokenizer, iteration_samples, args.include_rationale, lr=args.lr, epochs=1)
        model.save_pretrained(os.path.join(args.out_dir, f"model_iter_{iteration}"))
        tokenizer.save_pretrained(os.path.join(args.out_dir, f"model_iter_{iteration}"))

    engine.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Iterative SFT for verifier-only chess reasoning")
    parser.add_argument("--model_name", type=str, default="sshleifer/tiny-gpt2")
    parser.add_argument("--engine_path", type=str, default="stockfish")
    parser.add_argument("--depth", type=int, default=10)
    parser.add_argument("--multipv", type=int, default=4)
    parser.add_argument("--time_ms", type=int, default=0)
    parser.add_argument("--num_candidates", type=int, default=2)
    parser.add_argument("--iterations", type=int, default=1)
    parser.add_argument("--include_rationale", action="store_true")
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--out_dir", type=str, default="./data/output")
    parser.add_argument("--fens", nargs="*", help="Optional list of FEN strings")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    iterative_loop(args)


if __name__ == "__main__":
    main()
