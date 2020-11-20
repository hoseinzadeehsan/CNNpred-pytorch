from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch
import numpy as np

class WholeDataset(Dataset):

    def __init__(self, data, target):
        self.data = data
        self.target = target


    def __len__(self):
        return self.target.shape[0]

    def __getitem__(self, idx):

        return self.data[idx], self.target[idx]



def generate_batches(dataset,
                     batch_size,
                     shuffle=True,
                     drop_last=False,
                     device="cpu",
                     n_workers=0):
    dataloader = DataLoader(dataset=dataset,
                            batch_size=batch_size,
                            shuffle=shuffle,
                            drop_last=drop_last,
                            num_workers=n_workers,
                            pin_memory=False)

    for data, labels in dataloader:
        data = torch.unsqueeze(data, 1).float()
        labels = labels.float()
        yield data.to(device, non_blocking=True), labels.to(device, non_blocking=True)
