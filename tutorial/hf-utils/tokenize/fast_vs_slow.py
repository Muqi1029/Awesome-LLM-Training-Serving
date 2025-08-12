import os
import time

import matplotlib.pyplot as plt
from transformers import AutoTokenizer

MODEL_NAME = os.environ["QWEN330BA3BFP8"]
TEXTS = ["Hello world! This is a tokenizer speed test."] * 50000  # 大批量测试

# 加载 fast 和 slow 版本
tok_fast = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
tok_slow = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False)

# 测试分词结果一致性
sample_text = "I love Hugging Face!"
fast_tokens = tok_fast.tokenize(sample_text)
slow_tokens = tok_slow.tokenize(sample_text)

# 测速
start = time.time()
tok_fast(TEXTS)
fast_time = time.time() - start

start = time.time()
tok_slow(TEXTS)
slow_time = time.time() - start

# 打印基本信息
print("Fast tokenizer type:", type(tok_fast))
print("Slow tokenizer type:", type(tok_slow))
print("\nTokenize result (fast):", fast_tokens)
print("Tokenize result (slow):", slow_tokens)
print(f"\nFast tokenizer time: {fast_time:.4f} seconds")
print(f"Slow tokenizer time: {slow_time:.4f} seconds")

# 可视化对比
plt.bar(["Fast", "Slow"], [fast_time, slow_time], color=["green", "red"])
plt.ylabel("Time (seconds)")
plt.title("Fast vs Slow Tokenizer Speed")
plt.show()
