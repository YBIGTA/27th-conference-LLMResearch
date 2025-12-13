import json
import os
from typing import Dict, Iterable, List

from .schema import validate_sample_dict


def save_jsonl(path: str, samples: Iterable[Dict]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for sample in samples:
            f.write(json.dumps(sample) + "\n")


def append_jsonl(path: str, samples: Iterable[Dict]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        for sample in samples:
            f.write(json.dumps(sample) + "\n")


def load_jsonl(path: str) -> List[Dict]:
    data: List[Dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            validate_sample_dict(obj)
            data.append(obj)
    return data


__all__ = ["save_jsonl", "append_jsonl", "load_jsonl"]
