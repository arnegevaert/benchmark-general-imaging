from captum import attr
from torch import nn
import torch
from attribench import AttributionMethod


class Gradient(AttributionMethod):
    def __init__(self, model: nn.Module):
        self.method = attr.Saliency(model)

    def __call__(self, batch_x: torch.Tensor, batch_target: torch.Tensor):
        if not batch_x.requires_grad:
            batch_x.requires_grad = True
        return self.method.attribute(batch_x, target=batch_target, abs=False)
