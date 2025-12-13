import contextlib
import os
import shutil
from typing import Any, Dict, List, Optional

import chess
import chess.engine


class StockfishEngine:
    """Lightweight Stockfish wrapper for repeated analysis calls."""

    def __init__(
        self,
        engine_path: str = "stockfish",
        default_depth: int = 12,
        default_multipv: int = 1,
        default_time_ms: int = 0,
    ) -> None:
        self.engine_path = engine_path
        self.default_depth = default_depth
        self.default_multipv = default_multipv
        self.default_time_ms = default_time_ms
        self._engine: Optional[chess.engine.SimpleEngine] = None

    @contextlib.contextmanager
    def _ensure_engine(self) -> Any:
        engine_was_running = self._engine is not None
        if self._engine is None:
            if not shutil.which(self.engine_path):
                raise FileNotFoundError(f"Engine binary not found: {self.engine_path}")
            self._engine = chess.engine.SimpleEngine.popen_uci(self.engine_path)
        try:
            yield self._engine
        finally:
            if not engine_was_running:
                self.close()

    def analyze(
        self,
        fen: str,
        moves: Optional[List[str]] = None,
        multipv: Optional[int] = None,
        depth: Optional[int] = None,
        time_ms: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Run an engine analysis and return parsed multipv lines.

        Args:
            fen: Board state.
            moves: Optional list of root moves (UCI) to constrain search.
            multipv: Number of principal variations to request.
            depth: Search depth.
            time_ms: Search time in milliseconds.
        """
        multipv = multipv or self.default_multipv
        depth = depth or self.default_depth
        time_ms = time_ms if time_ms is not None else self.default_time_ms

        board = chess.Board(fen)
        root_moves = None
        if moves:
            try:
                root_moves = [chess.Move.from_uci(move) for move in moves]
            except ValueError as exc:
                raise ValueError(f"Invalid move in forced list: {moves}") from exc

        limits = chess.engine.Limit(depth=depth, time=time_ms / 1000 if time_ms else None)

        with self._ensure_engine() as engine:
            engine.configure({"MultiPV": multipv})
            info_list = engine.analyse(board, limits=limits, multipv=multipv, root_moves=root_moves)

        return [self._parse_info(info, board) for info in info_list]

    def refutation_pv(
        self,
        fen: str,
        move: str,
        depth: Optional[int] = None,
        time_ms: Optional[int] = None,
    ) -> List[str]:
        """Compute a short refutation PV for a specific move."""
        depth = depth or self.default_depth
        time_ms = time_ms if time_ms is not None else self.default_time_ms
        board = chess.Board(fen)
        try:
            board.push_uci(move)
        except ValueError:
            return []

        limits = chess.engine.Limit(depth=depth, time=time_ms / 1000 if time_ms else None)
        with self._ensure_engine() as engine:
            info = engine.analyse(board, limits=limits)

        pv = [uci for uci in self._extract_pv(info, board)]
        return pv[:3]

    def close(self) -> None:
        if self._engine:
            try:
                self._engine.quit()
            except chess.engine.EngineTerminatedError:
                pass
            self._engine = None

    def _parse_info(self, info: chess.engine.InfoDict, board: chess.Board) -> Dict[str, Any]:
        score = info.get("score")
        cp = None
        mate = None
        if score is not None:
            cp = score.pov(board.turn).score(mate_score=100000)
            mate = score.pov(board.turn).mate()

        pv_moves = list(self._extract_pv(info, board))
        wdl = None
        try:
            if score is not None and hasattr(score, "wdl"):
                wdl_score = score.pov(board.turn).wdl()
                wdl = [wdl_score.wins, wdl_score.draws, wdl_score.losses]
        except Exception:
            wdl = None

        return {
            "move": pv_moves[0] if pv_moves else None,
            "cp": cp,
            "wdl": wdl,
            "mate": mate,
            "pv": pv_moves,
        }

    def _extract_pv(self, info: chess.engine.InfoDict, board: chess.Board) -> List[str]:
        pv_seq: List[str] = []
        pv = info.get("pv")
        if not pv:
            return pv_seq
        temp_board = board.copy()
        for move in pv:
            if not temp_board.is_legal(move):
                break
            pv_seq.append(move.uci())
            temp_board.push(move)
        return pv_seq


__all__ = ["StockfishEngine"]
