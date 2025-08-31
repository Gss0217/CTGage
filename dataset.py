import torch
import numpy as np
from torch.utils.data import Dataset

class MyDataset(torch.utils.data.Dataset):
    def __init__(self, fhr, label, ids=None):
        self.fhr = torch.tensor(fhr, dtype=torch.float32)
        self.label = torch.tensor(label, dtype=torch.float32)
        self.ids = ids

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        if self.ids is not None:
            return self.fhr[idx], self.label[idx], self.ids[idx]
        else:
            return self.fhr[idx], self.label[idx]