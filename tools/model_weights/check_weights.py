# python check_weights.py --model-path THUDM/GLM-4-9B-0414
import argparse
import os
from glob import glob

from huggingface_hub import snapshot_download
from safetensors.torch import load_file

LINE_WIDTH = 110


def download_from_hub(model_path, cache_dir=None):
    # if cache_dir is None, use the default cache directory:
    # "${HF_HOME}/hub" = "~/.cache/huggingface/hub"
    if not os.path.isdir(model_path):
        dir_path = snapshot_download(model_path, cache_dir=cache_dir)
    else:
        dir_path = model_path
    print(f"\n📦 Model downloaded to: {dir_path}\n")

    total_params = 0
    num_weights = 0
    total_bytes = 0

    weight_files = glob(f"{dir_path}/*.safetensors")
    weight_files.sort()
    if args.max_checkpoints != -1:
        weight_files = weight_files[: args.max_checkpoints]

    for path in weight_files:
        print(f"\n🔍 Loading weights from: {path}")
        state_dict = load_file(path)
        # state_dict = safe_open(path, framework="pt", device="cpu")

        # Calculate total parameters
        num_params = sum(p.numel() for p in state_dict.values())
        total_params += num_params

        # Print summary header
        print("\n📊 Weight Summary:")
        print("-" * LINE_WIDTH)
        print(f"{'Layer Name':<60} {'Shape':<30} {'Dtype':<20}")
        print("-" * LINE_WIDTH)

        # Print each layer's information
        for key, value in state_dict.items():
            # key: str, value: Tensor
            total_bytes += value.numel() * value.element_size()
            print(f"{key:<60} {str(value.shape):<30} {str(value.dtype):<20}")
            num_weights += 1

        print("-" * LINE_WIDTH)
        print(f"parameters: {num_params:,}")

    print("Summarization of all weights".center(LINE_WIDTH, "-"))
    print(f"{'Total parameters:':<40} {total_params:<15,}")
    print(f"{'Num of weight files:':<40} {len(weight_files):<15}")
    print(f"{'Num of weights:':<40} {num_weights:<15}")
    print(f"{'Total bytes:':<40} {total_bytes / 1024 / 1024 / 1024:<15.2f} GB")
    print("-" * LINE_WIDTH)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--cache-dir", type=str, default=None)
    parser.add_argument("--max-checkpoints", type=int, default=-1)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    download_from_hub(args.model_path, args.cache_dir)
