# python check_weights.py --model-path THUDM/GLM-4-9B-0414
import argparse
import os
import re
from glob import glob

import torch
from huggingface_hub import snapshot_download
from safetensors.torch import load_file

MIN_NAME_WIDTH = 60
SHAPE_WIDTH = 24
DTYPE_WIDTH = 20
NUMEL_WIDTH = 16


def compile_regex(pattern):
    try:
        return re.compile(pattern)
    except re.error as exc:
        raise argparse.ArgumentTypeError(f"invalid regular expression: {exc}") from exc


def format_shape(value):
    if value.ndim == 0:
        return "scalar (0-D)"
    return str(tuple(value.shape))


def format_scalar(value):
    item = value.item()
    if isinstance(item, float):
        return f"{item:.9g}"
    if isinstance(item, complex):
        return f"{item:.9g}"
    return str(item)


def is_modelopt_nvfp4_weight(key, value, state_dict):
    """Detect ModelOpt NVFP4 weights without treating every uint8 tensor as FP4."""
    if value.dtype != torch.uint8 or value.ndim == 0 or not key.endswith(".weight"):
        return False

    prefix = key.removesuffix(".weight")
    return (
        f"{prefix}.weight_scale" in state_dict
        and f"{prefix}.weight_scale_2" in state_dict
    )


def tensor_details(key, value, state_dict):
    if value.ndim == 0:
        return f"value={format_scalar(value)}; scalar contains 1 element"

    if value.numel() == 0:
        return "EMPTY tensor: contains 0 elements"

    if is_modelopt_nvfp4_weight(key, value, state_dict):
        logical_shape = list(value.shape)
        logical_shape[-1] *= 2
        return (
            "ModelOpt NVFP4 packed: 2 FP4 values/uint8; "
            f"logical_shape={tuple(logical_shape)}"
        )

    return ""


def print_weight_summary(state_dict, name_filter=None):
    tensor_items = [
        (key, value)
        for key, value in state_dict.items()
        if isinstance(value, torch.Tensor)
    ]
    displayed_items = [
        (key, value)
        for key, value in tensor_items
        if name_filter is None or name_filter.search(key)
    ]

    displayed_rows = [
        (key, value, tensor_details(key, value, state_dict))
        for key, value in displayed_items
    ]
    name_width = max(
        MIN_NAME_WIDTH,
        max((len(key) for key, _ in displayed_items), default=len("Layer Name")),
    )
    details_width = max(
        len("Details"),
        max((len(details) for _, _, details in displayed_rows), default=0),
    )
    line_width = (
        name_width + SHAPE_WIDTH + DTYPE_WIDTH + NUMEL_WIDTH + details_width + 5
    )

    print("\n📊 Weight Summary:")
    print("-" * line_width)
    print(
        f"{'Layer Name':<{name_width}} "
        f"{'Shape':<{SHAPE_WIDTH}} "
        f"{'Dtype':<{DTYPE_WIDTH}} "
        f"{'Numel':>{NUMEL_WIDTH}}  Details"
    )
    print("-" * line_width)

    for key, value, details in displayed_rows:
        print(
            f"{key:<{name_width}} "
            f"{format_shape(value):<{SHAPE_WIDTH}} "
            f"{str(value.dtype):<{DTYPE_WIDTH}} "
            f"{value.numel():>{NUMEL_WIDTH},}  {details}"
        )

    print("-" * line_width)
    print(
        f"Stored tensor elements: {sum(value.numel() for _, value in tensor_items):,}"
    )
    if name_filter is not None:
        print(f"Displayed tensors: {len(displayed_items):,} / {len(tensor_items):,}")

    total_bytes = sum(value.numel() * value.element_size() for _, value in tensor_items)
    return len(tensor_items), total_bytes


def load_bin_file(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        # Compatibility with PyTorch versions that do not support weights_only.
        return torch.load(path, map_location="cpu")


def inspect_weight_files(weight_files, loader, name_filter=None):
    total_tensors = 0
    total_bytes = 0

    for path in weight_files:
        print(f"\n🔍 Loading weights from: {path}")
        state_dict = loader(path)
        num_tensors, file_bytes = print_weight_summary(state_dict, name_filter)
        total_tensors += num_tensors
        total_bytes += file_bytes

    return total_tensors, total_bytes


def download_from_hub(
    model_path,
    cache_dir=None,
    max_checkpoints=-1,
    name_filter=None,
):
    # If cache_dir is None, Hugging Face uses "${HF_HOME}/hub".
    if not os.path.isdir(model_path):
        dir_path = snapshot_download(model_path, cache_dir=cache_dir)
    else:
        dir_path = model_path
    print(f"\n📦 Model downloaded to: {dir_path}\n")

    weight_files = sorted(glob(f"{dir_path}/*.safetensors"))
    loader = load_file

    if not weight_files:
        weight_files = sorted(glob(f"{dir_path}/*.bin"))
        loader = load_bin_file

    if not weight_files:
        raise FileNotFoundError(
            f"No .safetensors or .bin weight files found in {dir_path}"
        )

    if max_checkpoints != -1:
        weight_files = weight_files[:max_checkpoints]

    total_tensors, total_bytes = inspect_weight_files(
        weight_files,
        loader,
        name_filter,
    )

    line_width = 110
    print("Summarization of inspected weights".center(line_width, "-"))
    print(f"{'Num of weight files:':<40} {len(weight_files):<15}")
    print(f"{'Num of tensors:':<40} {total_tensors:<15,}")
    print(f"{'Total storage bytes:':<40} {total_bytes:<15,}")
    print(f"{'Total storage size:':<40} {total_bytes / 1024**3:<15.2f} GiB")
    print("-" * line_width)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Inspect PyTorch or safetensors model weights."
    )
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--cache-dir", type=str, default=None)
    parser.add_argument("--max-checkpoints", type=int, default=-1)
    parser.add_argument(
        "--name-filter",
        type=compile_regex,
        default=None,
        help="Only display tensor names matching this regular expression.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    download_from_hub(
        args.model_path,
        cache_dir=args.cache_dir,
        max_checkpoints=args.max_checkpoints,
        name_filter=args.name_filter,
    )
