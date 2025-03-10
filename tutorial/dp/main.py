import time

import torch
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
        print(f"Time taken: {end - start} seconds")
        return result

    return wrapper


def main():
    ds = DummyDataset(num_examples=1000, weights=[1, 2, 3])
    dl = get_dataloader(ds)

    model = nn.Sequential(
        nn.Linear(3, 1),
    )

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    if torch.cuda.device_count() > 1:
        print(f"use {torch.cuda.device_count()} GPUs!")
        model = nn.DataParallel(model)

    model.train()

    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    for epoch in tqdm(range(10)):
        for inputs, targets in dl:
            optimizer.zero_grad()
            inputs = inputs.to(device)
            targets = targets.to(device)
            outputs = model(inputs)
            loss = F.mse_loss(outputs.squeeze(), targets)
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch} loss: {loss.item()}")

    print(model.module.state_dict())


if __name__ == "__main__":
    main()
