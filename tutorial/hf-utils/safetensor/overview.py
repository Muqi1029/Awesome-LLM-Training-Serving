from glob import glob
from pprint import pprint

import torch
from huggingface_hub import snapshot_download
from safetensors.torch import load_file
from torch import nn


def download_from_hub(model_name_or_path, cache_dir=None):
    # if cache_dir is None, use the default cache directory:
    # "${HF_HOME}/hub" = "~/.cache/huggingface/hub"
    dir_path = snapshot_download(model_name_or_path, cache_dir=cache_dir)
    for path in glob(f"{dir_path}/*.safetensors"):
        print(f"Loading {path}")
        state_dict = load_file(path)
        for key, value in state_dict.items():
            print(f"{key=}\t{value.shape=}\t{value.dtype=}")


class MyModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 10)
        # this module has 2 parameters: weight and bias
        # its name is "linear.weight" and "linear.bias"

        self.cust_param = nn.Parameter(torch.randn(10, 10))
        self.cust_param.load_weight = lambda: print("load_weight")

    def forward(self, x):
        return self.linear(x)


if __name__ == "__main__":
    download_from_hub("Qwen/Qwen3-0.6B")
    model = MyModule()

    params_dicts = dict(model.named_parameters())
    for name, param in params_dicts.items():
        print(f"{type(param)=}, {name=}\t{param.shape=}\t{param.dtype=}")
        if hasattr(param, "load_weight"):
            param.load_weight()

    # for name, param in model.named_parameters():
