from captum import attr
from torch import nn
import torch
from attrbench import AttributionMethod


class DeepLift(AttributionMethod):
    def __init__(self, model: nn.Module):
        self.method = attr.DeepLift(model)

    def __call__(self, batch_x: torch.Tensor, batch_target: torch.Tensor):
        return self.method.attribute(batch_x, target=batch_target)