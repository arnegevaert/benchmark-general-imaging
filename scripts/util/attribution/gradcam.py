from captum import attr
from torch import nn
import torch.nn.functional as F


class GradCAM:
    def __init__(self, model: nn.Module, last_conv_layer: nn.Module, relu_attributions=True):
        self.method = attr.LayerGradCam(model, last_conv_layer)
        self.relu_attributions = relu_attributions

    def __call__(self, x, target):
        # Compute attributions
        attrs = self.method.attribute(x, target, relu_attributions=self.relu_attributions)
        # Upsample attributions (this is equivalent to attr.LayerAttribution.interpolate,
        # but allows us to set align_corners to True)
        #upsampled = attr.LayerAttribution.interpolate(attrs, x.shape[-2:], interpolate_mode="bilinear")
        upsampled = F.interpolate(attrs, x.shape[-2:], mode="bilinear", align_corners=True)
        # GradCAM aggregates over channels, check if we need to duplicate attributions in order to match input shape
        if upsampled.shape[1] != x.shape[1]:
            upsampled = upsampled.repeat(1, x.shape[1], 1, 1)
        return upsampled
