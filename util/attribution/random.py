import torch


class Random:
    def __init__(self, positive_only=False):
        self.positive_only = positive_only

    def __call__(self, x, target):
        rand = torch.rand(*x.shape)
        return rand.to(x.device) if self.positive_only else (rand * 2 - 1).to(x.device)