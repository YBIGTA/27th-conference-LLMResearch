import dataclasses
from typing import Any, Dict, List, Optional


@dataclasses.dataclass
class EngineConfig:
    name: str
    depth: int
    multipv: int
    time_ms: int = 0


@dataclasses.dataclass
class EvalLine:
    move: str
    cp: Optional[int]
    wdl: Optional[List[int]]
    mate: Optional[int]
    pv: List[str]


@dataclasses.dataclass
class ContrastRationale:
    alt_move: Optional[str]
    refutation_pv: Optional[List[str]]
    note: Optional[str]


@dataclasses.dataclass
class Rationale:
    claim: Optional[str]
    evidence_pv: Optional[List[str]]
    contrast: Optional[ContrastRationale]
    verified: bool = False
    error_type: str = "NONE"


@dataclasses.dataclass
class Delta:
    cp: Optional[int]
    wdl_win_prob: Optional[float]


@dataclasses.dataclass
class Sample:
    id: str
    fen: str
    side_to_move: str
    candidates: List[str]
    engine: EngineConfig
    evals: List[EvalLine]
    chosen: Optional[str]
    rejected: Optional[str]
    delta: Optional[Delta]
    rationale: Optional[Rationale]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "fen": self.fen,
            "side_to_move": self.side_to_move,
            "candidates": self.candidates,
            "engine": dataclasses.asdict(self.engine),
            "evals": [dataclasses.asdict(e) for e in self.evals],
            "chosen": self.chosen,
            "rejected": self.rejected,
            "delta": dataclasses.asdict(self.delta) if self.delta else None,
            "rationale": rationale_to_dict(self.rationale),
        }


def rationale_to_dict(rationale: Optional[Rationale]) -> Optional[Dict[str, Any]]:
    if rationale is None:
        return None
    contrast = None
    if rationale.contrast:
        contrast = dataclasses.asdict(rationale.contrast)
    return {
        "claim": rationale.claim,
        "evidence_pv": rationale.evidence_pv,
        "contrast": contrast,
        "verified": rationale.verified,
        "error_type": rationale.error_type,
    }


def validate_sample_dict(sample: Dict[str, Any]) -> Dict[str, Any]:
    required_fields = ["id", "fen", "side_to_move", "candidates", "engine", "evals"]
    for field in required_fields:
        if field not in sample:
            raise ValueError(f"Missing field: {field}")
    if sample.get("side_to_move") not in {"w", "b"}:
        raise ValueError("side_to_move must be 'w' or 'b'")
    return sample


__all__ = [
    "Sample",
    "EngineConfig",
    "EvalLine",
    "Rationale",
    "ContrastRationale",
    "Delta",
    "validate_sample_dict",
]
