import re
from typing import Dict, List, Optional, Tuple

import chess


RATIONAL_FORMAT = re.compile(
    r"MOVE:\s*(?P<move>\S+)\s*" +
    r"CLAIM:\s*(?P<claim>.+?)\s*" +
    r"EVIDENCE_PV:\s*(?P<evidence>.+?)\s*" +
    r"CONTRAST_MOVE:\s*(?P<contrast_move>\S+)\s*" +
    r"CONTRAST_PV:\s*(?P<contrast_pv>.+)",
    re.DOTALL,
)


def parse_rationale_text(text: str) -> Dict[str, Optional[str]]:
    match = RATIONAL_FORMAT.search(text.strip())
    if not match:
        return {
            "move": None,
            "claim": None,
            "evidence_pv": None,
            "contrast_move": None,
            "contrast_pv": None,
        }
    return {
        "move": match.group("move").strip(),
        "claim": match.group("claim").strip(),
        "evidence_pv": match.group("evidence").strip(),
        "contrast_move": match.group("contrast_move").strip(),
        "contrast_pv": match.group("contrast_pv").strip(),
    }


def _is_legal_pv(board: chess.Board, moves: List[str]) -> bool:
    temp_board = board.copy()
    for mv in moves:
        try:
            move_obj = chess.Move.from_uci(mv)
        except ValueError:
            return False
        if not temp_board.is_legal(move_obj):
            return False
        temp_board.push(move_obj)
    return True


def verify_rationale(fen: str, rationale: Dict[str, Optional[str]]) -> Tuple[bool, str]:
    board = chess.Board(fen)
    evidence_moves = rationale.get("evidence_pv") or ""
    contrast_moves = rationale.get("contrast_pv") or ""
    evidence_seq = [mv for mv in evidence_moves.split() if mv]
    contrast_seq = [mv for mv in contrast_moves.split() if mv]

    if rationale.get("move"):
        try:
            move_obj = chess.Move.from_uci(rationale["move"])
        except ValueError:
            return False, "ILLEGAL"
        if not board.is_legal(move_obj):
            return False, "ILLEGAL"

    if evidence_seq and not _is_legal_pv(board, evidence_seq):
        return False, "RATIONALE_MISMATCH"

    if contrast_seq:
        try:
            temp_board = board.copy()
            if rationale.get("contrast_move"):
                temp_board.push_uci(rationale["contrast_move"])
        except ValueError:
            return False, "ILLEGAL"
        if not _is_legal_pv(temp_board, contrast_seq):
            return False, "RATIONALE_MISMATCH"

    if not rationale.get("claim"):
        return False, "RATIONALE_MISMATCH"
    return True, "NONE"


__all__ = ["parse_rationale_text", "verify_rationale"]
