import argparse
import math
import os
from typing import Dict, List, Optional

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, get_scheduler

from data.io import append_jsonl
from data.schema import EngineConfig
from engine.stockfish_wrapper import StockfishEngine
from rationale.verify import parse_rationale_text, verify_rationale


class PreferenceDataset(Dataset):
    def __init__(self, tokenizer, pairs: List[Dict[str, str]]):
        self.tokenizer = tokenizer
        self.pairs = pairs

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        pair = self.pairs[idx]
        prompt = pair["prompt"]
        chosen = pair["chosen"]
        rejected = pair["rejected"]

        chosen_ids = self.tokenizer(prompt + chosen, return_tensors="pt")
        rejected_ids = self.tokenizer(prompt + rejected, return_tensors="pt")

        return {
            "prompt": prompt,
            "chosen_ids": chosen_ids,
            "rejected_ids": rejected_ids,
        }


def build_prompt(fen: str, include_rationale: bool) -> str:
    if include_rationale:
        return (
            "You are a chess reasoning agent.\n"
            f"Given the position in FEN: {fen}\n"
            "Propose one move in UCI and a verifiable rationale.\n"
            "Output format:\nMOVE: ...\nCLAIM: ...\nEVIDENCE_PV: ...\nCONTRAST_MOVE: ...\nCONTRAST_PV: ...\n"
        )
    return (
        "You are a chess engine assistant.\n"
        f"Given the position in FEN: {fen}\n"
        "Output a single best move in UCI format.\nOutput format:\nMOVE: <uci>\n"
    )


def format_response(move: str, rationale: Optional[Dict[str, str]], include_rationale: bool) -> str:
    if include_rationale and rationale:
        return (
            f"MOVE: {move}\n"
            f"CLAIM: {rationale.get('claim', '')}\n"
            f"EVIDENCE_PV: {(rationale.get('evidence_pv') or '').strip()}\n"
            f"CONTRAST_MOVE: {rationale.get('contrast_move', '')}\n"
            f"CONTRAST_PV: {(rationale.get('contrast_pv') or '').strip()}\n"
        )
    return f"MOVE: {move}\n"


def logprob_response(model, tokenizer, prompt: str, response: str) -> torch.Tensor:
    prompt_ids = tokenizer(prompt, return_tensors="pt").to(model.device)
    full = tokenizer(prompt + response, return_tensors="pt").to(model.device)
    with torch.no_grad():
        logits = model(**full).logits
    log_probs = torch.log_softmax(logits, dim=-1)
    input_ids = full["input_ids"]
    prompt_len = prompt_ids["input_ids"].shape[1]
    resp_ids = input_ids[:, prompt_len:]
    resp_log_probs = log_probs[:, prompt_len - 1 : input_ids.shape[1] - 1, :]
    gathered = resp_log_probs.gather(2, resp_ids.unsqueeze(-1)).squeeze(-1)
    return gathered.sum(dim=-1)


def dpo_loss(model, tokenizer, pairs: List[Dict[str, str]], beta: float = 0.1) -> torch.Tensor:
    losses = []
    for pair in pairs:
        lp_chosen = logprob_response(model, tokenizer, pair["prompt"], pair["chosen"])
        lp_rejected = logprob_response(model, tokenizer, pair["prompt"], pair["rejected"])
        weight = pair.get("weight", 1.0)
        losses.append(-torch.logsigmoid(beta * (lp_chosen - lp_rejected)) * weight)
    return torch.stack(losses).mean()


def generate_pair(
    model,
    tokenizer,
    fen: str,
    engine: StockfishEngine,
    engine_cfg: EngineConfig,
    include_rationale: bool,
    cp_scale: float,
) -> Optional[Dict[str, str]]:
    prompt = build_prompt(fen, include_rationale)
    encoded = tokenizer(prompt, return_tensors="pt")
    output = model.generate(
        **{k: v for k, v in encoded.items()},
        max_new_tokens=64 if include_rationale else 8,
        do_sample=True,
        top_k=20,
        pad_token_id=tokenizer.eos_token_id,
    )
    text = tokenizer.decode(output[0], skip_special_tokens=True)
    parsed = parse_rationale_text(text)
    model_move = parsed.get("move")

    evals = engine.analyze(fen, multipv=engine_cfg.multipv, depth=engine_cfg.depth, time_ms=engine_cfg.time_ms)
    if not evals:
        return None
    best_eval = evals[0]
    chosen_move = best_eval.get("move")
    if not chosen_move or not model_move or chosen_move == model_move:
        return None

    model_eval = next((ev for ev in evals if ev.get("move") == model_move), None)
    chosen_cp = best_eval.get("cp")
    rejected_cp = model_eval.get("cp") if model_eval else None
    weight = 1.0
    if chosen_cp is not None and rejected_cp is not None and cp_scale > 0:
        weight = min(abs(chosen_cp - rejected_cp) / cp_scale, 1.0)
        if weight == 0:
            return None

    verified, error_type = verify_rationale(fen, parsed)
    rationale = parsed if verified else None

    pair = {
        "prompt": prompt,
        "chosen": format_response(chosen_move, rationale, include_rationale),
        "rejected": format_response(model_move, parsed, include_rationale),
        "weight": weight,
    }
    return pair


def iterative_dpo(args: argparse.Namespace) -> None:
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model_name)
    model.to(args.device)

    engine = StockfishEngine(
        engine_path=args.engine_path,
        default_depth=args.depth,
        default_multipv=args.multipv,
        default_time_ms=args.time_ms,
    )
    engine_cfg = EngineConfig(name="stockfish", depth=args.depth, multipv=args.multipv, time_ms=args.time_ms)

    os.makedirs(args.out_dir, exist_ok=True)
    for iteration in range(args.iterations):
        pairs: List[Dict[str, str]] = []
        for fen in args.fens:
            pair = generate_pair(
                model,
                tokenizer,
                fen,
                engine,
                engine_cfg,
                include_rationale=args.include_rationale,
                cp_scale=args.cp_scale,
            )
            if pair:
                pairs.append(pair)
        if not pairs:
            continue
        append_jsonl(os.path.join(args.out_dir, f"dpo_iter_{iteration}.jsonl"), pairs)
        dataset = PreferenceDataset(tokenizer, pairs)
        loader = DataLoader(dataset, batch_size=1, shuffle=True)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
        scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=len(loader) * args.epochs)

        model.train()
        for _ in range(args.epochs):
            for batch in loader:
                loss = dpo_loss(model, tokenizer, pairs, beta=args.beta)
                loss.backward()
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

        model.save_pretrained(os.path.join(args.out_dir, f"dpo_model_iter_{iteration}"))
        tokenizer.save_pretrained(os.path.join(args.out_dir, f"dpo_model_iter_{iteration}"))

    engine.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Iterative DPO with engine verifier")
    parser.add_argument("--model_name", type=str, default="sshleifer/tiny-gpt2")
    parser.add_argument("--engine_path", type=str, default="stockfish")
    parser.add_argument("--depth", type=int, default=10)
    parser.add_argument("--multipv", type=int, default=4)
    parser.add_argument("--time_ms", type=int, default=0)
    parser.add_argument("--iterations", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--include_rationale", action="store_true")
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--cp_scale", type=float, default=200.0)
    parser.add_argument("--fens", nargs="*", default=["rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"])
    parser.add_argument("--out_dir", type=str, default="./data/output")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    iterative_dpo(args)


if __name__ == "__main__":
    main()
