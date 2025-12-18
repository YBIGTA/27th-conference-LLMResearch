import os
import shutil
from typing import Any, Dict, List, Optional

import chess
import chess.engine


class StockfishEngine:
    """Lightweight Stockfish wrapper for repeated analysis calls."""

    # Common stockfish installation paths
    COMMON_PATHS = [
        "/usr/games/stockfish",           # Ubuntu/Debian apt install
        "/usr/bin/stockfish",             # Some Linux distros
        "/usr/local/bin/stockfish",       # Manual install
        "/opt/homebrew/bin/stockfish",    # macOS Homebrew (Apple Silicon)
        "/usr/local/Cellar/stockfish/*/bin/stockfish",  # macOS Homebrew (Intel)
    ]

    def __init__(
        self,
        engine_path: str = "stockfish",
        default_depth: int = 12,
        default_multipv: int = 1,
        default_time_ms: int = 0,
    ) -> None:
        self.engine_path = self._find_engine(engine_path)
        self.default_depth = default_depth
        self.default_multipv = default_multipv
        self.default_time_ms = default_time_ms
        self._engine: Optional[chess.engine.SimpleEngine] = None

    def _find_engine(self, engine_path: str) -> str:
        """Find stockfish binary, checking common paths if not in PATH."""
        # 1. If absolute path provided and exists, use it
        if os.path.isabs(engine_path) and os.path.isfile(engine_path):
            return engine_path
        
        # 2. Check if in PATH
        found = shutil.which(engine_path)
        if found:
            return found
        
        # 3. Check common installation paths
        for path in self.COMMON_PATHS:
            if os.path.isfile(path):
                return path
        
        # 4. Return original (will fail later with clear error)
        return engine_path

    def _ensure_engine(self) -> chess.engine.SimpleEngine:
        """Ensure engine is initialized and return it. Engine stays open for reuse."""
        if self._engine is None:
            if not os.path.isfile(self.engine_path) and not shutil.which(self.engine_path):
                raise FileNotFoundError(
                    f"Engine binary not found: {self.engine_path}\n"
                    f"Tried common paths: {self.COMMON_PATHS}\n"
                    f"Please install stockfish or specify --engine_path"
                )
            self._engine = chess.engine.SimpleEngine.popen_uci(self.engine_path)
        return self._engine

    def analyze(
        self,
        fen: str,
        moves: Optional[List[str]] = None,
        multipv: Optional[int] = None,
        depth: Optional[int] = None,
        time_ms: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
    
        multipv = multipv or self.default_multipv
        depth = depth or self.default_depth
        time_ms = time_ms if time_ms is not None else self.default_time_ms
    
        board = chess.Board(fen)
    
        root_moves = None
        if moves:
            root_moves = []
            for move in moves:
                try:
                    root_moves.append(chess.Move.from_uci(move))
                except ValueError:
                    continue
            if not root_moves:
                root_moves = None
    
        limits = chess.engine.Limit(
            depth=depth,
            time=(time_ms / 1000.0) if time_ms else None,
        )
    
        engine = self._ensure_engine()
        infos = engine.analyse(
            board,
            limit=limits,
            multipv=multipv,
            root_moves=root_moves,
        )
    
        if isinstance(infos, dict):
            infos = [infos]
    
        return [self._parse_info(info, board) for info in infos]

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
        engine = self._ensure_engine()
        info = engine.analyse(board, limit=limits)

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
