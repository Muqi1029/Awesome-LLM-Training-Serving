# python check_weights.py --model-path THUDM/GLM-4-9B-0414
import argparse
from glob import glob

from huggingface_hub import snapshot_download
from safetensors.torch import load_file


def download_from_hub(model_path, cache_dir=None):
    # if cache_dir is None, use the default cache directory:
    # "${HF_HOME}/hub" = "~/.cache/huggingface/hub"
    dir_path = snapshot_download(model_path, cache_dir=cache_dir)
    print(f"\n📦 Model downloaded to: {dir_path}\n")

    total_params = 0
    num_weights = 0
    total_bytes = 0

    weight_files = glob(f"{dir_path}/*.safetensors")
    weight_files.sort()

    for path in weight_files:
        print(f"\n🔍 Loading weights from: {path}")
        state_dict = load_file(path)

        # Calculate total parameters
        num_params = sum(p.numel() for p in state_dict.values())
        total_params += num_params

        # Print summary header
        print("\n📊 Weight Summary:")
        print("-" * 100)
        print(f"{'Layer Name':<50} {'Shape':<30} {'Dtype':<20}")
        print("-" * 100)

        # Print each layer's information
        for key, value in state_dict.items():
            # key: str, value: Tensor
            total_bytes += value.numel() * value.element_size()
            print(f"{key:<50} {str(value.shape):<30} {str(value.dtype):<20}")
            num_weights += 1

        print("-" * 100)
        print(f"parameters: {num_params:,}")

    print("Summarization of all weights".center(100, "-"))
    print(f"{'Total parameters:':<30} {total_params:<15,}")
    print(f"{'Num of weight files:':<30} {len(weight_files):<15}")
    print(f"{'Num of weights:':<30} {num_weights:<15}")
    print(f"{'Total bytes:':<30} {total_bytes / 1024 / 1024 / 1024:<15.2f} GB")
    print("-" * 100)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    download_from_hub(args.model_path)
