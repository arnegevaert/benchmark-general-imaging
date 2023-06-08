from torch import nn
from captum import attr
from util.attribution import GradCAM
import warnings


class GuidedGradCAM:
    def __init__(self, model: nn.Module):
        self.gc = GradCAM(model)
        self.gbp = attr.GuidedBackprop(model)

    def __call__(self, x, target):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # Compute GBP attributions
            gbp_attrs = self.gbp.attribute(x, target)
            # Compute attributions
            gc_attrs = self.gc(x, target)
            return gbp_attrs * gc_attrs