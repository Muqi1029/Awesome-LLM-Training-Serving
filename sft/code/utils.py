import random

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_model_and_tokenizer(config):
    tokenizer = AutoTokenizer.from_pretrained(config["model_name_or_path"])
    model = AutoModelForCausalLM.from_pretrained(
        config["model_name_or_path"], torch_dtype="bfloat16"
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    return model, tokenizer


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
