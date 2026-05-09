#!/usr/bin/env bash

CUDA_VISIBLE_DEVICES=0,1 accelerate launch code/acc.py
