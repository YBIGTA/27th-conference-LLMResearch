"""Skeleton for Stage 2 self-play and post-mortem STaR analysis."""
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class GameOutcome:
    moves: List[str]
    fens: List[str]
    regrets: List[float]
    annotations: Optional[List[str]] = None


def run_self_play(model, tokenizer, num_games: int = 1):
    raise NotImplementedError("Self-play loop should pit the current policy against itself or a weaker opponent.")


def post_mortem_star(game: GameOutcome):
    raise NotImplementedError("Analyze blunders with Stockfish and generate rationales for high-regret moves.")
