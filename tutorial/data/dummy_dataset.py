import torch
from torch.utils.data import Dataset


class DummyDataset(Dataset):
    def __init__(self, num_examples: int, weights: list[float], sigma=0.1):
        self.weights = weights
        self.inputs = torch.randn((num_examples, len(weights)))
        self.targets = (
            self.inputs @ torch.tensor(weights, dtype=torch.float32)
            + torch.randn(num_examples) * sigma
        )

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx]
