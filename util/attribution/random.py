import torch
from attrbench import AttributionMethod
from torch import nn


class Random(AttributionMethod):
    def __init__(self, model: nn.Module, positive_only=False):
        super().__init__(model)
        self.positive_only = positive_only

    def __call__(self, x, target):
        rand = torch.rand(*x.shape)
        return (
            rand.to(x.device)
            if self.positive_only
            else (rand * 2 - 1).to(x.device)
        )
