import torch
from attribench import AttributionMethod
from torch import nn
from captum import attr
from util.attribution import GradCAM
import warnings


class GuidedGradCAM(AttributionMethod):
    def __init__(self, model: nn.Module):
        self.gc = GradCAM(model)
        self.gbp = attr.GuidedBackprop(model)

    def __call__(self, batch_x: torch.Tensor, batch_target: torch.Tensor):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # Compute GBP attributions
            gbp_attrs = self.gbp.attribute(batch_x, batch_target)
            # Compute attributions
            gc_attrs = self.gc(batch_x, batch_target)
            return gbp_attrs * gc_attrs