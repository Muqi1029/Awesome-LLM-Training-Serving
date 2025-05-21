# script: torchrun --nproc_per_node=#{world_size} group_list.py
import argparse
import os
from typing import Dict, List

import torch
import torch.distributed as dist

_groups: Dict[str, dist.ProcessGroup] = {}


def _register_group(name: str, group: dist.ProcessGroup):
    """Register a process group with a name for future reference"""
    _groups[name] = group


def main(group_ranks: List[List[int]], backend, device):
    # Initialize the default process group
    dist.init_process_group(backend=backend)

    try:
        rank = dist.get_rank()  # global rank
        local_rank = int(os.environ["LOCAL_RANK"])  # local rank

        # Create and register process groups
        for i, ranks in enumerate(group_ranks):
            group_name = f"group-{i}"
            try:
                device_group = dist.new_group(ranks, backend=backend)
                _register_group(group_name, device_group)

                if rank in ranks:
                    world_size = len(ranks)
                    rank_in_group = ranks.index(rank)
                    print(
                        f"{local_rank=}, {rank=}, group={i}, {world_size=}, {rank_in_group=}"
                    )

                    # Perform all_gather operation within the group
                    data = torch.tensor([rank], dtype=torch.int64, device=device)
                    # Use empty instead of zeros for better performance
                    tensor_list = [
                        torch.empty_like(data, device=device) for _ in range(world_size)
                    ]

                    dist.all_gather(tensor_list, data, group=device_group)

                    # Add barrier for synchronization
                    dist.barrier(group=device_group)

                    if rank_in_group == 0:
                        print(f"gather result of {group_name}: {tensor_list}")
            except RuntimeError as e:
                print(f"Error in group {i}: {e}")
    finally:
        # Ensure process group is properly destroyed
        dist.destroy_process_group()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--use-gpu", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.use_gpu:
        if not torch.cuda.is_available():
            raise ValueError("GPU is not available, please use CPU!")

        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)

        device = torch.device("cuda")
        backend = "nccl"
    else:
        device = torch.device("cpu")
        backend = "gloo"

    group_ranks = [[0, 1, 2], [3, 4]]
    # in this case, we need to run the script with 5 processes
    # script: torchrun --nproc_per_node=5 group_list.py [--use-gpu]

    main(group_ranks, backend, device)
