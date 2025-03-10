from torch.utils.data import DataLoader


def get_dataloader(dataset, batch_size=16, shuffle=True):
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
