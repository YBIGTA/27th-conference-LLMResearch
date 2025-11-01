# /root/ReTool-RL/preprocess.py
from datasets import load_dataset, DatasetDict

PROMPT_KEYS = ["prompt", "question", "problem", "instruction", "input", "query"]
RESP_KEYS   = ["response", "answer", "solution", "output"]

def pick_case_insensitive(row: dict, candidates):
    """row의 키를 소문자 맵으로 만들어 대소문자 무시 탐색"""
    lower_map = {k.lower(): k for k in row.keys()}
    for c in candidates:
        if c in lower_map:
            return row[lower_map[c]]
    return None

def to_messages(x):
    # 이미 [{"role":"user","content":...}] 형태면 그대로 사용
    if isinstance(x, list) and x and isinstance(x[0], dict) and "content" in x[0]:
        return x
    return [{"role":"user","content": str(x)}]

def convert(repo_id, out_dir, splits=None, num_proc=1):
    # splits=None이면 dataset이 가진 모든 split 사용
    dsdict: DatasetDict = load_dataset(repo_id)  # 예: {'train': Dataset} 또는 {'train','validation',...}
    if splits is None:
        splits = list(dsdict.keys())

    for split in splits:
        ds = dsdict[split]

        def _map(row):
            # 1) prompt 후보에서(대/소문자 무시) 텍스트 가져오기
            ptxt = pick_case_insensitive(row, PROMPT_KEYS)
            if ptxt is None:
                # 예: AIME_2024 -> 'Problem'이 실제 키인데 대소문자 처리 필요
                raise KeyError(f"{repo_id}::{split}: no prompt-like key in {list(row.keys())}")
            out = {"prompt": to_messages(ptxt)}

            # 2) 응답(정답/해설)도 있으면 보존(평가/분석용)
            rtxt = pick_case_insensitive(row, RESP_KEYS)
            if rtxt is not None:
                out["response"] = str(rtxt)

            # 선택적으로 원본 id 보존(있으면)
            for k in row.keys():
                if k.lower() in ["id", "idx"]:
                    out["example_id"] = row[k]
                    break

            return out

        keep = ["prompt", "response", "example_id"]
        ds2 = ds.map(_map,
                     remove_columns=[c for c in ds.column_names if c not in keep],
                     num_proc=num_proc)
        path = f"{out_dir}/{split}.parquet"
        ds2.to_parquet(path)
        print("wrote", path)

if __name__ == "__main__":
    # AIME_2024 : 컬럼 ['ID','Problem','Solution','Answer'] 지원됨
    convert("Maxwell-Jia/AIME_2024", "/root/27th-conference-LLMResearch/data/Maxwell-Jia/AIME_2024-rl", num_proc=1)
    # DAPO-Math-17k
    convert("BytedTsinghua-SIA/DAPO-Math-17k", "/root/27th-conference-LLMResearch/data/BytedTsinghua-SIA/DAPO-Math-17k-rl", num_proc=1)
    # AIME_2025
    convert("yentinglin/aime_2025", "/root/27th-conference-LLMResearch/data/yentinglin/aime_2025-rl", num_proc=1)
