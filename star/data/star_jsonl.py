import dataclasses
import json
from pathlib import Path
from typing import Iterable, List, Optional


@dataclasses.dataclass
class StarSample:
    fen: str
    move: str
    rationale: str
    eval_best: Optional[float] = None
    eval_model: Optional[float] = None
    regret: Optional[float] = None
    source: str = "success"
    position_id: Optional[str] = None

    def to_json(self) -> str:
        return json.dumps(dataclasses.asdict(self), ensure_ascii=False)


class StarJsonlWriter:
    def __init__(self, path: str):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, records: Iterable[StarSample]) -> None:
        with self.path.open("a", encoding="utf-8") as f:
            for rec in records:
                f.write(rec.to_json() + "\n")


def load_star_jsonl(path: str) -> List[StarSample]:
    samples: List[StarSample] = []
    json_path = Path(path)
    if not json_path.exists():
        return samples
    with json_path.open("r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            samples.append(StarSample(**data))
    return samples


__all__ = ["StarSample", "StarJsonlWriter", "load_star_jsonl"]
