from captum import attr
import torch
from torch import nn
from attribench import AttributionMethod


class InputXGradient(AttributionMethod):
    def __init__(self, model: nn.Module):
        self.method = attr.InputXGradient(model)

    def __call__(self, batch_x: torch.Tensor, batch_target: torch.Tensor):
        return self.method.attribute(batch_x, target=batch_target)
