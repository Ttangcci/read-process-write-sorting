import torch
from torch.utils.data import Dataset


class SortingDataset(Dataset):
    def __init__(self, num_samples=10000, set_size=5):
        self.num_samples = num_samples
        self.set_size = set_size

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # x 是一个无序集合，形状为 [set_size, 1]；target 是从小到大排序后的原始下标。
        x = torch.rand(self.set_size, 1)
        target = torch.argsort(x.squeeze(-1), dim=0)
        return x, target
