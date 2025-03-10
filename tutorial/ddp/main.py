import os
import time

import hydra
import torch
import torch.distributed as dist
from omegaconf import DictConfig
from safetensors.torch import save_file
from torch import nn
from torch.distributed import destroy_process_group, init_process_group
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from tutorial.data.dummy_dataset import DummyDataset


def compute_time(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(f"Time taken: {(end - start):.4f} seconds")
        return result

    return wrapper


def cleanup():
    destroy_process_group()


@compute_time
def train(model, epochs, optimizer, dl, device):
    for epoch in tqdm(range(epochs)):
        epoch_loss = 0
        for inputs, targets in dl:
            optimizer.zero_grad()
            inputs = inputs.to(device)
            targets = targets.to(device)
            outputs = model(inputs)
            loss = F.mse_loss(outputs.squeeze(), targets)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        # Convert epoch_loss to tensor and move to correct device
        epoch_loss_tensor = torch.tensor(epoch_loss, device=device)
        dist.all_reduce(epoch_loss_tensor)
        if dist.get_rank() == 0:
            print(f"Epoch {epoch} loss: {epoch_loss_tensor.item() / len(dl)}")


@hydra.main(version_base=None, config_path=".", config_name="train")
def main(config: DictConfig):
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    print(
        f"rank: {rank}, world_size: {world_size}, MASTER_ADDR: {os.environ['MASTER_ADDR']}, MASTER_PORT: {os.environ['MASTER_PORT']}"
    )
    torch.cuda.set_device(rank)
    init_process_group(backend="nccl", rank=rank, world_size=world_size)

    ds = DummyDataset(
        num_examples=config.dataset.num_examples, weights=config.dataset.weight
    )
    dl = DataLoader(
        ds, batch_size=config.training.batch_size, sampler=DistributedSampler(ds)
    )

    model = nn.Sequential(nn.Linear(len(config.dataset.weight), 1)).to(rank)
    model = DDP(model, device_ids=[rank])
    optimizer = torch.optim.SGD(model.parameters(), lr=config.training.lr)

    model.train()
    train(model, config.training.epochs, optimizer, dl, rank)

    # make sure all processes have finished training
    dist.barrier()
    if rank == 0:
        if torch.cuda.device_count() > 1:
            print(model.module.state_dict())
        else:
            print(model.state_dict())
        save_file(model.state_dict(), "model.safetensors")

    destroy_process_group()


if __name__ == "__main__":
    main()
