import contextlib
import dataclasses
from typing import List, Optional, Tuple

import chess
import chess.engine


@dataclasses.dataclass
class EngineEval:
    fen: str
    best_move: Optional[str]
    best_cp: float
    best_mate: Optional[int]
    candidate_move: Optional[str]
    candidate_cp: float
    candidate_mate: Optional[int]

    @property
    def regret(self) -> float:
        return compute_regret(self.best_cp, self.candidate_cp, self.best_mate, self.candidate_mate)


def score_to_cp(score: chess.engine.PovScore) -> Tuple[float, Optional[int]]:
    mate = score.mate()
    if mate is not None:
        direction = 1 if mate > 0 else -1
        return 100000.0 * direction - direction * abs(mate), mate
    cp = score.score(mate_score=100000)
    return float(cp), None


def compute_regret(best_cp: float, candidate_cp: float, best_mate: Optional[int], candidate_mate: Optional[int]) -> float:
    if best_mate is not None or candidate_mate is not None:
        normalized_best = best_cp if best_mate is None else 100000.0 * (1 if best_mate > 0 else -1)
        normalized_candidate = candidate_cp if candidate_mate is None else 100000.0 * (1 if candidate_mate > 0 else -1)
        return normalized_best - normalized_candidate
    return best_cp - candidate_cp


class StockfishEngine:
    def __init__(self, engine_path: str, depth: int = 18, multipv: int = 4, time_limit_ms: int = 0):
        self.engine_path = engine_path
        self.depth = depth
        self.multipv = multipv
        self.time_limit_ms = time_limit_ms
        self.engine: Optional[chess.engine.SimpleEngine] = None

    def __enter__(self):
        self.engine = chess.engine.SimpleEngine.popen_uci(self.engine_path)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if self.engine:
            with contextlib.suppress(Exception):
                self.engine.quit()

    def analyze(self, fen: str, forced_move: Optional[str] = None) -> EngineEval:
        if self.engine is None:
            raise RuntimeError("Engine not initialized. Use context manager or call .__enter__().")

        board = chess.Board(fen)
        limit = self._build_limit()
        info_list = self.engine.analyse(board, limit, multipv=self.multipv)
        best_info = info_list[0] if isinstance(info_list, list) else info_list
        best_score = best_info["score"].pov(board.turn)
        best_cp, best_mate = score_to_cp(best_score)
        best_move = best_info.get("pv", [None])[0]
        best_move_uci = best_move.uci() if best_move else None

        candidate_cp, candidate_mate, candidate_move_uci = best_cp, best_mate, best_move_uci
        if forced_move:
            candidate_move = chess.Move.from_uci(forced_move)
            if candidate_move not in board.legal_moves:
                raise ValueError(f"Illegal move {forced_move} for position {fen}")
            board.push(candidate_move)
            forced_info = self.engine.analyse(board, limit, multipv=1)
            forced_data = forced_info[0] if isinstance(forced_info, list) else forced_info
            forced_score = forced_data["score"].pov(board.turn)
            candidate_cp, candidate_mate = score_to_cp(forced_score)
            candidate_move_uci = forced_move

        return EngineEval(
            fen=fen,
            best_move=best_move_uci,
            best_cp=best_cp,
            best_mate=best_mate,
            candidate_move=candidate_move_uci,
            candidate_cp=candidate_cp,
            candidate_mate=candidate_mate,
        )

    def _build_limit(self) -> chess.engine.Limit:
        if self.time_limit_ms and self.time_limit_ms > 0:
            return chess.engine.Limit(time=self.time_limit_ms / 1000.0)
        return chess.engine.Limit(depth=self.depth)


__all__: List[str] = ["StockfishEngine", "EngineEval", "score_to_cp", "compute_regret"]
