from torch import nn
from captum import attr
from experiments.lib.attribution import GradCAM


class GuidedGradCAM:
    def __init__(self, model: nn.Module, last_conv_layer: nn.Module, relu_attributions=True):
        self.gc = GradCAM(model, last_conv_layer, relu_attributions)
        self.gbp = attr.GuidedBackprop(model)
        self.relu_attributions = relu_attributions

    def __call__(self, x, target):
        # Compute GBP attributions
        gbp_attrs = self.gbp.attribute(x, target)
        # Compute attributions
        gc_attrs = self.gc(x, target)
        return gbp_attrs * gc_attrs