import argparse
import logging
import random
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import yaml
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)

from star.data.star_jsonl import StarJsonlWriter, StarSample
from star.engine.stockfish_wrapper import EngineEval, StockfishEngine

logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger(__name__)


FEN_REGEX = re.compile(r"([prnbqkPRNBQK1-8]+\/[prnbqkPRNBQK1-8]+\/[prnbqkPRNBQK1-8]+\/[prnbqkPRNBQK1-8]+\/[prnbqkPRNBQK1-8]+\/[prnbqkPRNBQK1-8]+\/[prnbqkPRNBQK1-8]+\/[prnbqkPRNBQK1-8]+\s+[wb]\s+(?:-|[KQkq]+)\s+(?:-|[a-h][36])\s+[0-9]+\s+[0-9]+)")
UCI_REGEX = re.compile(r"([a-h][1-8][a-h][1-8][qrbn]?)", re.IGNORECASE)


class AdaptiveSampler:
    def __init__(self, target_success_rate: float, min_visits: int, replay_size: int):
        self.target = target_success_rate
        self.min_visits = min_visits
        self.replay_size = replay_size
        self.stats: Dict[str, Dict[str, float]] = {}

    def record(self, position_id: str, success: bool) -> None:
        stat = self.stats.setdefault(position_id, {"visits": 0, "success": 0})
        stat["visits"] += 1
        stat["success"] += float(success)
        if len(self.stats) > self.replay_size:
            oldest_key = next(iter(self.stats))
            del self.stats[oldest_key]

    def should_sample(self, position_id: str) -> bool:
        stat = self.stats.get(position_id)
        if not stat or stat["visits"] < self.min_visits:
            return True
        success_rate = stat["success"] / max(stat["visits"], 1)
        if success_rate < self.target:
            return True
        retain_prob = max(0.1, 1.0 - (success_rate - self.target))
        return random.random() < retain_prob


def load_config(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
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


def load_model_and_tokenizer(model_config: Dict):
    quant_config = None
    if model_config.get("load_in_4bit"):
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=getattr(torch, model_config.get("torch_dtype", "float16")),
        )
    tokenizer = AutoTokenizer.from_pretrained(model_config["name_or_path"])
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_config["name_or_path"],
        quantization_config=quant_config,
        torch_dtype=getattr(torch, model_config.get("torch_dtype", "float16")),
        device_map="auto",
    )
    return model, tokenizer


def extract_fen(text: str) -> Optional[str]:
    match = FEN_REGEX.search(text)
    return match.group(1) if match else None


def parse_uci(text: str) -> Optional[str]:
    move_match = None
    for line in text.splitlines():
        if line.lower().startswith("move"):
            move_match = UCI_REGEX.search(line)
            break
    if move_match is None:
        move_match = UCI_REGEX.search(text)
    if not move_match:
        return None
    return move_match.group(1).lower()


def format_prompt(fen: str) -> str:
    return (
        "You are a chess engine. Given the FEN, propose the best move and briefly explain why.\n"
        f"FEN: {fen}\n"
        "Respond with:\nMOVE: <uci>\nRATIONALE: <text>\n"
    )


def generate_candidates(
    model, tokenizer, fen: str, k_samples: int, temperature: float, top_p: float, max_new_tokens: int
) -> List[Tuple[str, str]]:
    prompt = format_prompt(fen)
    encoded = tokenizer(prompt, return_tensors="pt").to(model.device)
    candidates: List[Tuple[str, str]] = []
    attempts = 0
    max_attempts = max(k_samples * 3, k_samples + 2)
    while len(candidates) < k_samples and attempts < max_attempts:
        attempts += 1
        outputs = model.generate(
            **encoded,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.eos_token_id,
        )
        decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
        move = parse_uci(decoded)
        if move:
            rationale_section = decoded.split("RATIONALE:")
            rationale = rationale_section[1].strip() if len(rationale_section) > 1 else ""
            candidates.append((move, rationale))
    return candidates


def load_positions(config: Dict) -> List[Tuple[str, Optional[str]]]:
    positions: List[Tuple[str, Optional[str]]] = []
    mate_root = config["data"].get("mate_root", "dataset/data_mate")
    mate_train = Path(mate_root) / "train_stripped.jsonl"
    if not mate_train.exists():
        alt = Path("star") / Path(mate_root)
        mate_train = alt / "train_stripped.jsonl"
    if mate_train.exists():
        dataset = load_dataset("json", data_files=str(mate_train))["train"]
        for row in dataset:
            fen = extract_fen(row.get("question", ""))
            if fen:
                positions.append((fen, row.get("answer")))
    unlabeled_path = config["data"].get("unlabeled_fen_path")
    if unlabeled_path:
        with open(unlabeled_path, "r", encoding="utf-8") as f:
            for line in f:
                fen = line.strip()
                if fen:
                    positions.append((fen, None))
    return positions


def select_successes(
    evals: List[EngineEval],
    threshold_cp: float,
    accept_max: int,
    rationales: List[str],
    position_id: str,
) -> List[StarSample]:
    accepted: List[StarSample] = []
    for engine_eval, rationale in zip(evals, rationales):
        if engine_eval.regret <= threshold_cp:
            accepted.append(
                StarSample(
                    fen=engine_eval.fen,
                    move=engine_eval.candidate_move or "",
                    rationale=rationale,
                    eval_best=engine_eval.best_cp,
                    eval_model=engine_eval.candidate_cp,
                    regret=engine_eval.regret,
                    source="success",
                    position_id=position_id,
                )
            )
        if len(accepted) >= accept_max:
            break
    return accepted


def rationalize_failure(
    model, tokenizer, engine_eval: EngineEval, position_id: str, max_new_tokens: int, temperature: float, top_p: float
) -> StarSample:
    prompt = (
        "The engine's best move is provided. Explain the plan and why this move is strong.\n"
        f"FEN: {engine_eval.fen}\n"
        f"BEST MOVE: {engine_eval.best_move}\n"
        "RATIONALE:"
    )
    encoded = tokenizer(prompt, return_tensors="pt").to(model.device)
    output = model.generate(
        **encoded,
        do_sample=True,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        pad_token_id=tokenizer.eos_token_id,
    )
    decoded = tokenizer.decode(output[0], skip_special_tokens=True)
    rationale = decoded.split("RATIONALE:")[-1].strip()
    return StarSample(
        fen=engine_eval.fen,
        move=engine_eval.best_move or "",
        rationale=rationale,
        eval_best=engine_eval.best_cp,
        eval_model=engine_eval.candidate_cp,
        regret=engine_eval.regret,
        source="fallback",
        position_id=position_id,
    )


def run_star_stage1(config: Dict) -> None:
    model, tokenizer = load_model_and_tokenizer(config["model"])
    positions = load_positions(config)
    adaptive_cfg = config.get("adastar", {})
    adaptive_sampler: Optional[AdaptiveSampler] = None
    if adaptive_cfg.get("enable", True):
        adaptive_sampler = AdaptiveSampler(
            target_success_rate=adaptive_cfg.get("target_success_rate", 0.5),
            min_visits=adaptive_cfg.get("min_visits", 2),
            replay_size=adaptive_cfg.get("replay_size", 200000),
        )
    writer = StarJsonlWriter(config["data"]["output_star_jsonl"])
    engine_cfg = config.get("engine", {})
    star_cfg = config.get("star", {})
    k_samples = star_cfg.get("k_samples", 4)
    accept_max = star_cfg.get("accept_max_per_position", 1)
    regret_threshold = star_cfg.get("regret_threshold_cp", 30)

    with StockfishEngine(
        engine_path=engine_cfg["path"],
        depth=engine_cfg.get("depth", 18),
        multipv=engine_cfg.get("multipv", 4),
        time_limit_ms=engine_cfg.get("time_limit_ms", 0),
    ) as engine:
        for idx, (fen, _) in enumerate(positions):
            position_id = f"pos_{idx}"
            if adaptive_sampler and not adaptive_sampler.should_sample(position_id):
                continue
            candidates = generate_candidates(
                model,
                tokenizer,
                fen,
                k_samples=k_samples,
                temperature=star_cfg.get("temperature", 0.8),
                top_p=star_cfg.get("top_p", 0.95),
                max_new_tokens=star_cfg.get("max_new_tokens", 128),
            )
            evals: List[EngineEval] = []
            rationales: List[str] = []
            for move, rationale in candidates:
                try:
                    engine_eval = engine.analyze(fen, forced_move=move)
                    evals.append(engine_eval)
                    rationales.append(rationale)
                except Exception as exc:
                    LOGGER.warning("Engine failed on %s with move %s: %s", fen, move, exc)
            successes = select_successes(evals, regret_threshold, accept_max, rationales, position_id)
            if successes:
                writer.append(successes)
                if adaptive_sampler:
                    adaptive_sampler.record(position_id, True)
                continue
            if adaptive_sampler:
                adaptive_sampler.record(position_id, False)
            if not star_cfg.get("fallback_rationalize_on_fail", True):
                continue
            if not evals:
                LOGGER.warning("No valid candidates for %s", fen)
                continue
            best_eval = max(evals, key=lambda ev: ev.best_cp)
            fallback = rationalize_failure(
                model,
                tokenizer,
                best_eval,
                position_id=position_id,
                max_new_tokens=star_cfg.get("max_new_tokens", 128),
                temperature=star_cfg.get("temperature", 0.8),
                top_p=star_cfg.get("top_p", 0.95),
            )
            writer.append([fallback])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Stage 1 STaR loop for chess")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    args, unknown = parser.parse_known_args()
    args.overrides = unknown
    return args


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    if getattr(args, "overrides", None):
        config = apply_overrides(config, args.overrides)
    run_star_stage1(config)


if __name__ == "__main__":
    main()
