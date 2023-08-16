from torch.utils.data import Dataset
import torch


class OpenMLDataset(Dataset):
    def __init__(self, data, target, pred_type):
        self.data = data
        self.target = target
        self.pred_type = pred_type
        assert self.pred_type in ["classification", "regression"]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return torch.tensor(self.data[idx], dtype=torch.float32), torch.tensor(
            self.target[idx],
            dtype=torch.long
            if self.pred_type == "classification"
            else torch.float32,
        )
