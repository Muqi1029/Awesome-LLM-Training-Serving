#!/usr/bin/env bash

# deepspeed --include localhost:0,1,2,3 sft_qwen_alpaca/train.py
deepspeed --num_gpus 4 sft_qwen_alpaca/train.py

# torchrun --nproc_per_node=4 sft_qwen_alpaca/train.py
