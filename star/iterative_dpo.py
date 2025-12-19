import argparse
import json
import math
import os
import re
from typing import Dict, List, Optional

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, get_scheduler

from data.io import append_jsonl
from data.schema import EngineConfig
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


def get_batch_logprobs(
    model,
    tokenizer,
    prompts: List[str],
    responses: List[str],
) -> torch.Tensor:
    """
    배치 단위로 response의 log probability 계산.
    DPO 논문 구현 참고: https://arxiv.org/abs/2305.18290
    """
    batch_logprobs = []
    
    for prompt, response in zip(prompts, responses):
        # 토큰화
        prompt_ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
        full_ids = tokenizer(prompt + response, return_tensors="pt", add_special_tokens=True)
        
        prompt_ids = {k: v.to(model.device) for k, v in prompt_ids.items()}
        full_ids = {k: v.to(model.device) for k, v in full_ids.items()}
        
        prompt_len = prompt_ids["input_ids"].shape[1]
        
        # Forward pass (gradient 필요)
        outputs = model(**full_ids)
        logits = outputs.logits
        
        # Log softmax
        log_probs = F.log_softmax(logits, dim=-1)
        
        # Response 부분의 log prob만 추출
        # logits[i]는 token[i+1]을 예측하므로, prompt_len-1부터 시작
        input_ids = full_ids["input_ids"]
        resp_ids = input_ids[:, prompt_len:]  # response tokens
        
        # 각 response token의 log prob 추출
        resp_log_probs = log_probs[:, prompt_len - 1 : -1, :]  # [batch, resp_len, vocab]
        
        # 실제 token에 해당하는 log prob gather
        token_log_probs = resp_log_probs.gather(2, resp_ids.unsqueeze(-1)).squeeze(-1)
        
        # 합산 (sequence-level log prob)
        batch_logprobs.append(token_log_probs.sum(dim=-1))
    
    return torch.cat(batch_logprobs)


def dpo_loss(
    model, 
    tokenizer, 
    pairs: List[Dict[str, str]], 
    beta: float = 0.1,
    reference_free: bool = True,
) -> torch.Tensor:
    """
    DPO (Direct Preference Optimization) Loss 계산.
    
    논문: https://arxiv.org/abs/2305.18290
    L_DPO = -E[log σ(β * (log π(y_w|x) - log π(y_l|x)))]
    
    Args:
        model: 학습 중인 모델
        tokenizer: 토크나이저
        pairs: [{"prompt": ..., "chosen": ..., "rejected": ..., "weight": ...}, ...]
        beta: KL penalty 계수 (기본 0.1)
        reference_free: True면 reference model 없이 학습 (간단한 버전)
    """
    if not pairs:
        return torch.tensor(0.0, device=next(model.parameters()).device, requires_grad=True)
    
    total_loss = torch.tensor(0.0, device=next(model.parameters()).device, requires_grad=True)
    
    for pair in pairs:
        prompt = pair["prompt"]
        chosen = pair["chosen"]
        rejected = pair["rejected"]
        weight = pair.get("weight", 1.0)
        
        # Chosen과 rejected의 log probability 계산
        chosen_logprob = get_batch_logprobs(model, tokenizer, [prompt], [chosen])
        rejected_logprob = get_batch_logprobs(model, tokenizer, [prompt], [rejected])
        
        # DPO loss: -log σ(β * (log π(chosen) - log π(rejected)))
        logits_diff = beta * (chosen_logprob - rejected_logprob)
        loss = -F.logsigmoid(logits_diff) * weight
        
        total_loss = total_loss + loss.mean()
    
    return total_loss / len(pairs)


def generate_moves_batch(
    model,
    tokenizer,
    fens: List[str],
    include_rationale: bool,
    batch_size: int = 8,
) -> List[str]:
    """배치 단위로 모델 추론 수행 (훨씬 빠름)"""
    import time
    
    all_texts = []
    prompts = [build_prompt(fen, include_rationale) for fen in fens]
    
    # 디버그: 모델 위치 확인
    print(f"[DEBUG] Model device: {next(model.parameters()).device}")
    print(f"[DEBUG] Model dtype: {next(model.parameters()).dtype}")
    print(f"[DEBUG] Total prompts: {len(prompts)}, batch_size: {batch_size}")
    
    model.eval()
    with torch.no_grad():
        for i in range(0, len(prompts), batch_size):
            batch_prompts = prompts[i:i + batch_size]
            
            t0 = time.time()
            encoded = tokenizer(
                batch_prompts, 
                return_tensors="pt", 
                padding=True,
                truncation=True,
            ).to(model.device)
            t1 = time.time()
            
            print(f"[DEBUG] Batch {i//batch_size}: input_ids shape={encoded['input_ids'].shape}, tokenize time={t1-t0:.2f}s")
            
            t2 = time.time()
            outputs = model.generate(
                **encoded,
                max_new_tokens=32 if include_rationale else 8,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
                use_cache=True,
            )
            t3 = time.time()
            
            print(f"[DEBUG] Generate time: {t3-t2:.2f}s, output shape={outputs.shape}")
            
            # 배치 디코딩
            texts = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            all_texts.extend(texts)
    
    return all_texts


def extract_move_simple(text: str) -> Optional[str]:
    """단순 MOVE: xxx 형식에서 move 추출"""
    match = re.search(r"MOVE:\s*(\S+)", text)
    if match:
        move = match.group(1).strip()
        # <uci> 같은 템플릿이 아닌 실제 수인지 확인 (예: e2e4, b1c3)
        if re.match(r'^[a-h][1-8][a-h][1-8][qrbn]?$', move):
            return move
    
    # 2. 마지막 줄에서 UCI 패턴 찾기 (few-shot 프롬프트 후 바로 수가 나오는 경우)
    lines = text.strip().split('\n')
    for line in reversed(lines):
        # UCI 수 패턴: a-h + 1-8 + a-h + 1-8 + optional promotion (예: e2e4, e7e8q)
        uci_match = re.search(r'\b([a-h][1-8][a-h][1-8][qrbn]?)\b', line)
        if uci_match:
            return uci_match.group(1)
    
    return None


def generate_pair_from_text(
    fen: str,
    model_text: str,
    engine: StockfishEngine,
    engine_cfg: EngineConfig,
    include_rationale: bool,
    cp_scale: float,
) -> Optional[Dict[str, str]]:
    """미리 생성된 텍스트로부터 pair 생성"""
    prompt = build_prompt(fen, include_rationale)
    
    # rationale 모드면 전체 파싱, 아니면 move만 추출
    if include_rationale:
        parsed = parse_rationale_text(model_text)
        model_move = parsed.get("move")
    else:
        parsed = {"move": None, "claim": None, "evidence_pv": None, "contrast_move": None, "contrast_pv": None}
        model_move = extract_move_simple(model_text)

    # Stockfish 분석
    evals = engine.analyze(fen, multipv=engine_cfg.multipv, depth=engine_cfg.depth, time_ms=engine_cfg.time_ms)
    if not evals or len(evals) < 2:
        return None
    
    best_eval = evals[0]
    chosen_move = best_eval.get("move")
    if not chosen_move:
        return None
    
    # 모델이 유효한 수를 생성하지 못했으면, Stockfish의 2번째 수를 rejected로 사용
    if not model_move or model_move == chosen_move:
        # 차선 수를 rejected로 사용 (첫 iteration 부트스트랩)
        second_eval = evals[1]
        model_move = second_eval.get("move")
        if not model_move or model_move == chosen_move:
            return None
        rejected_cp = second_eval.get("cp")
    else:
        model_eval = next((ev for ev in evals if ev.get("move") == model_move), None)
        rejected_cp = model_eval.get("cp") if model_eval else None
    
    chosen_cp = best_eval.get("cp")
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


def generate_pair(
    model,
    tokenizer,
    fen: str,
    engine: StockfishEngine,
    engine_cfg: EngineConfig,
    include_rationale: bool,
    cp_scale: float,
) -> Optional[Dict[str, str]]:
    """단일 FEN 처리 (하위 호환성 유지)"""
    texts = generate_moves_batch(model, tokenizer, [fen], include_rationale, batch_size=1)
    if not texts:
        return None
    return generate_pair_from_text(fen, texts[0], engine, engine_cfg, include_rationale, cp_scale)


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
            print(f"[DPO] Model loaded from checkpoint: {model_path}")
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
            print(f"[DPO] Model loaded from HuggingFace: {args.model_name}")
    
    # Move to device if not distributed (distributed handles this internally)
    if not distributed:
        device = getattr(args, 'device', 'cuda')  # 기본값 cuda로 변경
        if device == 'cuda' and not torch.cuda.is_available():
            print("[WARNING] CUDA not available, falling back to CPU")
            device = 'cpu'
        print(f"[DPO] Moving model to {device}...")
        model = model.to(device)
        print(f"[DPO] Model is now on: {next(model.parameters()).device}")
    
    return model, tokenizer


def iterative_dpo(args: argparse.Namespace) -> None:
    """Main DPO training loop (called via iterative_cli.py or directly)."""
    distributed = getattr(args, 'distributed', False) and getattr(args, 'world_size', 1) > 1
    rank = getattr(args, '_rank', 0)  # Set by fsdp_main for distributed
    
    # Load model and tokenizer using utils.py (get_loaded_model_tokenizer or get_model_tokenizer)
    model, tokenizer = load_model_and_tokenizer(args, rank=rank, distributed=distributed)

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
    batch_size = getattr(args, 'batch_size', 8)  # 배치 크기 (기본 8)
    
    for iteration in range(args.iterations):
        pairs: List[Dict[str, str]] = []
        total_fens = len(fens)
        print(f"[DPO Iter {iteration}] Generating moves for {total_fens} FENs (batch_size={batch_size})...")
        
        # 1단계: 배치로 모든 모델 출력 생성 (빠름!)
        all_texts = generate_moves_batch(
            model, tokenizer, fens, 
            include_rationale=args.include_rationale,
            batch_size=batch_size,
        )
        print(f"  -> Model inference done. Processing with Stockfish...")
        
        # 2단계: Stockfish로 pair 생성
        for i, (fen, text) in enumerate(zip(fens, all_texts)):
            pair = generate_pair_from_text(
                fen, text, engine, engine_cfg,
                include_rationale=args.include_rationale,
                cp_scale=args.cp_scale,
            )
            if pair:
                pairs.append(pair)
            # 진행률 10%마다 출력
            if (i + 1) % max(1, total_fens // 10) == 0:
                print(f"  [{i+1}/{total_fens}] pairs: {len(pairs)}")
        if not pairs:
            continue
        append_jsonl(os.path.join(args.out_dir, f"dpo_iter_{iteration}.jsonl"), pairs)
        dataset = PreferenceDataset(tokenizer, pairs)
        loader = DataLoader(dataset, batch_size=1, shuffle=True)
        
        # Optimizer setup (similar to get_optimizer_scheduler_step_based in utils.py)
        optimizer_type = getattr(args, 'optimizer', 'AdamW')
        weight_decay = getattr(args, 'weight_decay', 0.01)
        warm_up_steps = getattr(args, 'warm_up_steps', 0)
        
        if optimizer_type == "AdamW":
            optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=weight_decay)
        elif optimizer_type == "Adam":
            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        else:
            optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=weight_decay)
        
        # Scheduler with warm-up (from utils.py pattern)
        total_steps = len(loader) * args.epochs
        def lr_lambda(current_step: int):
            if current_step < warm_up_steps:
                return float(current_step) / float(max(1, warm_up_steps))
            return 1.0
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

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
        
        # Use log_args from utils.py for cumulative logging
        metrics_log_path = os.path.join(args.out_dir, "dpo_metrics_log.json")
        log_args(metrics_log_path, **metrics)
        
        # Also save individual iteration metrics
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
    parser.add_argument("--optimizer", type=str, default="AdamW", help="Optimizer type: AdamW or Adam (from utils.py pattern)")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay for optimizer")
    parser.add_argument("--warm_up_steps", type=int, default=0, help="Number of warmup steps for scheduler")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--precision", type=str, default="bf16", help="Model precision: bf16 or fp16")
    parser.add_argument("--distributed", action="store_true", help="Enable distributed training with DDP/FSDP (uses utils.py setup/cleanup)")
    parser.add_argument("--world_size", type=int, default=1, help="Number of GPUs for distributed training")
    parser.add_argument("--lora", type=json.loads, default=None, help="LoRA config as JSON string, e.g. '{\"lora_rank\": 32, \"lora_alpha\": 64, \"lora_dropout\": 0.1}'")
    parser.add_argument("--cp_scale", type=float, default=200.0)
    parser.add_argument("--base_model_path", type=str, default=None, help="Path to base model checkpoint (file or directory). If None, loads from model_name.")
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


def fsdp_main(rank: int, world_size: int, args: argparse.Namespace) -> None:
    """Entry point for distributed training (called by mp.spawn)."""
    setup(rank, world_size)
    torch.cuda.set_device(rank)
    
    # Set distributed flag
    args.distributed = True
    args.world_size = world_size
    args._rank = rank  # Internal use for distributed
    
    try:
        iterative_dpo(args)
    finally:
        cleanup()


def main() -> None:
    """Direct execution entry point (supports --distributed flag)."""
    args = parse_args()
    
    distributed = getattr(args, 'distributed', False) and getattr(args, 'world_size', 1) > 1
    
    if distributed:
        # Distributed training using utils.py setup/cleanup
        print(f"[DPO] Starting distributed training with {args.world_size} GPUs")
        mp.spawn(
            fsdp_main,
            args=(args.world_size, args),
            nprocs=args.world_size,
            join=True,
        )
    else:
        # Single GPU training (same as iterative_cli.py call)
        iterative_dpo(args)


if __name__ == "__main__":
    main()
