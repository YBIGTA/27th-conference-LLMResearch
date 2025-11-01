#!/bin/bash
set -x

nnodes=1
nproc_per_node=4
master_addr=${master_addr:-127.0.0.1}
master_port=${master_port:-29500}
node_rank=${ARNOLD_ID:-0}

project_name=retool
experiment_name=multiturn-sft-qwen-2.5-3b-instruct

HDFS_ROOT=${HDFS_ROOT:-$PWD}
DATA_ROOT=${DATA_ROOT:-$PWD}

TRAIN_DATA=/root/27th-conference-LLMResearch/ReTool/ReTool-SFT/data/train-00000-of-00001.parquet
EVAL_DATA=/root/27th-conference-LLMResearch/ReTool/ReTool-SFT/data/train-00000-of-00001.parquet
MODEL_PATH=/root/verl/model/Qwen2.5-3B-Instruct
SAVE_PATH=/root/27th-conference-LLMResearch/ReTool/ReTool-SFT/model

torchrun --nnodes=$nnodes \
     --nproc_per_node=$nproc_per_node \
     --master-addr=$master_addr \
     --master-port=$master_port \
     --node-rank=$node_rank \
     -m verl.trainer.fsdp_sft_trainer \
    data.train_files=$TRAIN_DATA \
    data.val_files=$EVAL_DATA \
    data.max_length=16384 \
    data.train_batch_size=32 \
    data.multiturn.enable=true \
    data.multiturn.messages_key=messages \
    data.multiturn.tools_key=tools \
    data.micro_batch_size_per_gpu=2 \
    model.partial_pretrain=$MODEL_PATH \
    model.strategy=fsdp \
    trainer.default_local_dir=$SAVE_PATH \
    trainer.project_name=wuxibin-multiturn-sft \
    trainer.experiment_name=$experiment_name \
    trainer.logger='["console","wandb"]' \
    trainer.total_epochs=2 \
    trainer.save_freq=62 \
    ulysses_sequence_parallel_size=4 \
    use_remove_padding=true