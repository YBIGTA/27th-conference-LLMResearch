import argparse
import json
import math
import os
import re
from typing import Dict, List, Optional

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, get_scheduler

from star.data.io import append_jsonl
from star.data.schema import EngineConfig
from star.engine.stockfish_wrapper import StockfishEngine
from star.rationale.verify import parse_rationale_text, verify_rationale

ROOT_DIR = os.path.dirname(__file__)
DEFAULT_MATE_TRAIN = os.path.join(ROOT_DIR, "datasets", "data_mate", "train_stripped.jsonl")
DEFAULT_MATE_TEST = os.path.join(ROOT_DIR, "datasets", "data_mate", "test_stripped.jsonl")


def _extract_fen(question: str) -> Optional[str]:
    match = re.search(r"board is \"([^\"]+)\"", question)
    return match.group(1) if match else None


def load_mate_fens(path: str) -> List[str]:
    fens: List[str] = []
    if not path or not os.path.exists(path):
        return fens
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            record = json.loads(line)
            question = record.get("question", "")
            fen = _extract_fen(question)
            if fen:
                fens.append(fen)
    return fens


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


def evaluate_model(
    model,
    tokenizer,
    fens: List[str],
    engine: StockfishEngine,
    engine_cfg: EngineConfig,
    include_rationale: bool,
) -> Dict[str, float]:
    """Measure simple move agreement and average cp gap versus engine best moves."""

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

    fens = args.fens or load_mate_fens(args.mate_train_path)
    if not fens:
        raise ValueError("No FENs provided and data_mate could not be loaded")

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
        for fen in fens:
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
        metrics_path = os.path.join(args.out_dir, f"dpo_metrics_iter_{iteration}.json")
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)
        print(f"[DPO Iteration {iteration}] train metrics: {train_metrics}")
        if test_metrics:
            print(f"[DPO Iteration {iteration}] test metrics: {test_metrics}")

        model.save_pretrained(os.path.join(args.out_dir, f"dpo_model_iter_{iteration}"))
        tokenizer.save_pretrained(os.path.join(args.out_dir, f"dpo_model_iter_{iteration}"))

    engine.close()


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Iterative DPO with engine verifier")
    parser.add_argument("--config", type=str, default=None, help="Path to JSON config with hyperparameters")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-3B")
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
    parser.add_argument("--fens", nargs="*", help="Optional list of FEN strings; defaults to data_mate train split")
    parser.add_argument("--eval_fens", nargs="*", help="Optional held-out FEN strings for evaluation")
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
    parser.add_argument("--out_dir", type=str, default="./data/output")
    known_args, _ = parser.parse_known_args(argv)
    if known_args.config:
        with open(known_args.config, "r", encoding="utf-8") as f:
            config = json.load(f)
        parser.set_defaults(**config)
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    iterative_dpo(args)


if __name__ == "__main__":
    main()
