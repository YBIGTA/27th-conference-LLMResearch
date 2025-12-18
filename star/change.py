import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# 1. 베이스 아키텍처 이름 (SFT 전에 쓰던 원본 모델)
BASE_NAME = "Qwen/Qwen2.5-3B"

# 2. .pth 체크포인트 경로
CKPT_PATH = "/root/27th-conference-LLMResearch/star/LLM_Research/checkpoints/final_model.pth"

# 3. HF 포맷으로 저장할 디렉터리
SAVE_DIR = "/root/27th-conference-LLMResearch/star/LLM_Research_hf"

print(f"Loading base model: {BASE_NAME}")
model = AutoModelForCausalLM.from_pretrained(BASE_NAME, torch_dtype="auto")

print(f"Loading checkpoint: {CKPT_PATH}")
state = torch.load(CKPT_PATH, map_location="cpu")

# state_dict 형태에 따라 키 정리
if isinstance(state, dict):
    if "state_dict" in state:
        state = state["state_dict"]
    elif "model" in state:
        state = state["model"]

# module. prefix 제거 등 필요할 수 있음
new_state = {}
for k, v in state.items():
    new_k = k
    if new_k.startswith("module."):
        new_k = new_k[len("module."):]
    new_state[new_k] = v

print("Applying checkpoint to base model (load_state_dict)...")
missing, unexpected = model.load_state_dict(new_state, strict=False)
print("Missing keys:", len(missing))
print("Unexpected keys:", len(unexpected))

print(f"Saving HF-format model to: {SAVE_DIR}")
model.save_pretrained(SAVE_DIR)

print("Saving tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(BASE_NAME, use_fast=True)
tokenizer.save_pretrained(SAVE_DIR)

print("Done.")
