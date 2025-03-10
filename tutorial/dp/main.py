import time

import hydra
import torch
from omegaconf import DictConfig
from torch import nn
from torch.nn import functional as F
from tqdm import tqdm

from data.dummy_dataset import DummyDataset
from tutorial.utils import get_dataloader


def compute_time(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(f"Time taken: {(end - start):.4f} seconds")
        return result

    return wrapper


@compute_time
def train(model, epochs, optimizer, dl, device):
    model.train()
    for epoch in tqdm(range(epochs)):
        for inputs, targets in dl:
            optimizer.zero_grad()
            inputs = inputs.to(device)
            targets = targets.to(device)
            outputs = model(inputs)
            loss = F.mse_loss(outputs.squeeze(), targets)
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch} loss: {loss.item()}")


@hydra.main(version_base=None, config_path=".", config_name="train")
def main(config: DictConfig):
    ds = DummyDataset(
        num_examples=config.dataset.num_examples, weights=config.dataset.weight
    )
    dl = get_dataloader(ds)

    model = nn.Sequential(nn.Linear(len(config.dataset.weight), 1))

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    if torch.cuda.device_count() > 1:
        print(f"use {torch.cuda.device_count()} GPUs!")
        model = nn.DataParallel(model)
    else:
        print("use single GPU")

    optimizer = torch.optim.SGD(model.parameters(), lr=config.training.lr)

    train(model, config.training.epochs, optimizer, dl, device)

    if torch.cuda.device_count() > 1:
        print(model.module.state_dict())
    else:
        print(model.state_dict())


if __name__ == "__main__":
    main()
