from captum import attr
import warnings
import torch
from torch import nn
from attribench import AttributionMethod


class GuidedBackprop(AttributionMethod):
    def __init__(self, model: nn.Module):
        self.method = attr.GuidedBackprop(model)

    def __call__(self, batch_x: torch.Tensor, batch_target: torch.Tensor):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return self.method.attribute(batch_x, target=batch_target)
