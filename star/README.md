# AdaSTaR: Adaptive Sampling in Training Self-Taught Reasoners

## Contents

- [Project Structure](#Project-Structure)
- [Set up & Installation](#Set-up-&-Installation)
- [Run an experiment](#run-an-experiment)
- [Reference](#reference)


# Project Structure
```
AdaSTaR/
├── configs/                          # Configuration files for AdaSTaR (e.g., hyperparameters, experiment setups)
│
├── configs_method/                   # Configuration files for Adastar
│
├── datasets/                         # Datasets used for this project
│
├── n_shot_prompts/                   # Prompt files for few-shot (N-shot) learning setups
│
├── create_finetune_tfrecords.py      # Script for generating TF Record-format data used in AdaSTaR training
├── device_inference_adastar_new_square.py  # Inference script for AdaSTaR using a modified square-based structure
├── device_inference_adastar_new.py   # Standard inference script for AdaSTaR stochastic version
├── device_inference.py               # General-purpose inference script (specifically for evaluation)
├── device_train.py                   # Main training script for both AdaSTaR and its stochastic version
├── iteration_train.py                # Script for iterative training loop setups
├── README.md                         # Main documentation file
├── requirements.txt                  # List of required Python packages
├── utils_adastar.py                  # Utility functions specific to AdaSTaR
└── utils.py/                         # Utility functions for general STaR flow
```
This repository implements AdaSTaR and its stochastic version, supporting both training and inference workflows.
More detailed instructions on how to set up and run experiments are provided in the sections below.


## Set up & Installation

```bash
# Clone the repository
git clone
# Install dependencies
pip install -r requirements.txt
```


## Run an experiment

### AdaSTaR
```bash
python iteration_train.py --config=configs/example.json --method=adastar_new_square --seed=10
```

### AdaSTaR - Stochastic Version
```bash
python iteration_train.py --config=configs/example.json --method=adastar_new --seed=10
```

Several predefined configuration files are available under the `configs/` directory. Each config file is named after the dataset or model it corresponds to.  
You can select the appropriate config file depending on the dataset or model you want to use.

The naming convention is as follows:

- `{dataset}.json`: uses the **Llama-3.2-3B** model
- `{dataset}_qwen.json`: uses the **Qwen2.5-3B** model
- `{dataset}7b.json`: uses the **Gemma-7B** model

### Available Datasets

The following datasets are currently supported:

- `anli_r1`
- `arc_challenge`
- `cladder`
- `cqa` (CommonsenseQA)
- `gsm8k`
- `svamp`

### Model Support per Dataset

| Dataset       | Llama-3.2-3B | Qwen2.5-3B | Gemma-7B |
|---------------|:------------:|:----------:|:--------:|
| anli_r1       | ✅          | ❌         | ✅      |
| arc_challenge | ✅          | ✅         | ✅      |
| cladder       | ✅          | ❌         | ❌      |
| cqa           | ✅          | ❌         | ❌      |
| gsm8k         | ✅          | ✅         | ❌      |
| svamp         | ✅          | ✅         | ✅      |

✅ **LLaMA 3B** supports **all datasets**.  
✅ **Qwen 3B** supports: `arc_challenge`, `gsm8k`, `svamp`  
✅ **Gemma 7B** supports: `anli_r1`, `arc_challenge`, `svamp`  

❌ in the table simply means that a config file has not been provided yet.  
You can still use that dataset with the model by creating a compatible config file manually.  
Detailed instructions on how to write a config file are provided in the section below.


## Configuration File Structure

Each experiment is configured via a JSON file located in the `configs/` directory.  
These configuration files define the dataset, model type, training parameters, and more.

A typical config file for 3B models (e.g., LLaMA 3B, Qwen 3B) looks like this:

```json
{
  "epochs": 1,
  "grad_accumulation": 1,
  "gen_length": 381,
  "batch_size": 2,
  "test_batch_size": 32,
  "lr": 1e-05,
  "weight_decay": 0.01,
  "warm_up_steps": 100,
  "model_dir": "checkpoints/",
  "log_divisor": 100,
  "save_divisor": 5,
  "exp_name": "testrun",
  "optimizer": "Adam",
  "scheduler": "linear",
  "precision": "bf16",
  "model_name": "meta-llama/Llama-3.2-3B",
  "max_length": 512,
  "n_shot": 6,
  "self_consistency": 0,
  "delete_model_after_loading": true,
  "accumulate": false,
  "task": "arc_challenge",
  "inference_temp": 1.0,
  "no_hint": false,
  "base_model_path" : null
}
```

For 7B models (e.g., Gemma 7B), an additional lora section is required:
```json
{
  ...
  "lora": {
    "lora_rank": 32,
    "lora_alpha": 64,
    "lora_dropout": 0.1
  }
}
```

If you're creating a new config for a model-dataset combination not included by default,
use one of the existing files as a template and modify it accordingly.

## Log File Storage

Logs are saved inside the corresponding method's directory using the following structure:
```
{dataset_name}/{experiment_name}/eval_log.json
```

- `{dataset_name}`: the name of the dataset used  
- `{experiment_name}`: a unique identifier in the format:  
  `{dataset_name}_{method_name}_{seed}`

#### Example

If you're using the `gsm8k` dataset with the `adastar_new_square` method and seed `10`, the log file will be saved at:
```
gsm8k/gsm8k_adastar_new_square_10/eval_log.json
```
These logs contain evaluation results, and are useful for comparing performance across different methods, datasets.

## Engine-guided STaR for chess (proposed pipeline)

The repository can also serve as a template for extending STaR/AdaSTaR to chess settings where a strong engine (e.g., Stockfish) is available as an oracle. The pipeline below turns large unlabeled position pools into rationale-rich supervision and then scales to full-game self-play.

### Stage 1 – STaR(+AdaSTaR) for engine-aligned move prediction

1. **Unlabeled pool**: Crawl tens of millions of positions from sources such as lichess or public FEN databases; only the `position` string is required.
2. **Model sampling**: For each position, sample `k` trajectories from the current policy \(\pi_\theta\), each containing a reasoning trace and a proposed move.
3. **Engine evaluation**: Use Stockfish to score the best move (`eval_best`) and the model move (`eval_model`), and compute `regret = eval_best - eval_model` (centipawns or win-probability delta).
4. **Success selection**: Treat candidates with `regret` below a threshold (e.g., 20–30 cp) as successful STaR rationales; append their `(position, reasoning, move, eval, regret)` tuples to the training set with high weight.
5. **Failure rationalization**: If no candidate clears the threshold, ask the model to explain Stockfish’s `best_move` (“why this move is strong tactically/strategically”) and store that rationale as auxiliary supervision.
6. **Adaptive sampling (AdaSTaR)**: Maintain per-position win-rate/regret stats. Downsample trivial positions and prioritize borderline cases that the model often misses to form a curriculum.

Iterating this loop steadily tunes \(\pi_\theta\) toward engine-consistent moves while keeping rationales grounded in verifiable quality.

### Stage 2 – Self-play with post-mortem STaR

1. **Game generation**: Run full games via self-play (both sides as \(\pi_\theta\)), vs. a weaker checkpoint, or vs. a small engine.
2. **Stockfish review**: After the game, compute per-move `eval_before`/`eval_after` and derive a regret curve to identify inaccuracies, mistakes, and blunders.
3. **Post-mortem rationales**: At each blunder point, query “what was the better plan here?” using the engine best line as the answer, or resample model moves and keep those with low regret; capture accompanying strategic/tactical explanations.
4. **Quality-weighted updates**: Treat the analyzed trajectories as offline RL data (AWR/AWAC style), weighting moves by negative regret so the policy optimizes overall win rate and engine-aligned decisions.

This two-stage recipe removes human labeling from the loop while steadily improving both move selection and explanatory quality through engine-verifiable feedback.

## Reference

If you find this work helpful, please consider citing our paper:
```bibtex
@inproceedings{koh2025adastar,
  title={AdaSTaR: Adaptive Data Sampling for Training Self-Taught Reasoners},
  author={Koh, Woosung and Oh, Wonbeen and Jang, Jaein and Lee, MinHyung and Kim, Hyeongjin and Kim, Ah Yeon and Kim, Joonkee and Lee, Junghyun and Kim, Taehyeon and Yun, Se-Young},
  booktitle={The Thirty-ninth Annual Conference on Neural Information Processing Systems},
  year={2025},
  url={https://openreview.net/forum?id=D6PwC6Xogv}
}
```

## Chess STaR Stage 1 (engine-guided)

The `configs/stage1_star.yaml` file sets up the Qwen2.5-3B model with Stockfish evaluation, adaptive sampling, and
generation/output paths pinned to `dataset/data_mate`.

Run the STaR data generation loop:

```
python -m star.stage1_star_loop --config star/configs/stage1_star.yaml
```

Run baseline vs. STaR SFT fine-tuning:

```
# Baseline SFT without rationales
python -m star.train_star_sft --config star/configs/stage1_star.yaml train.mode=baseline_sft

# STaR-augmented SFT with success/fallback rationales
python -m star.train_star_sft --config star/configs/stage1_star.yaml train.mode=star_sft
```

The generated STaR samples are written to `dataset/generated/star_stage1.jsonl` using the schema defined in
`star/data/star_jsonl.py`. The Stockfish wrapper lives in `star/engine/stockfish_wrapper.py` and handles regret computation
with mate-aware scoring.
