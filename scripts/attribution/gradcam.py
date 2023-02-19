from captum import attr
from torch import nn
import torch.nn.functional as F
import torch
from attrbench import AttributionMethod


class GradCAM(AttributionMethod):
    def __init__(self, model: nn.Module):
        last_conv_layer = model.get_last_conv_layer()
        self.method = attr.LayerGradCam(model, last_conv_layer)

    def __call__(self, batch_x: torch.Tensor, batch_target: torch.Tensor):
        # Compute attributions
        attrs = self.method.attribute(batch_x, batch_target,
                                      relu_attributions=True)
        # Upsample attributions (this is equivalent to attr.LayerAttribution.interpolate,
        # but allows us to set align_corners to True)
        #upsampled = attr.LayerAttribution.interpolate(attrs, x.shape[-2:], interpolate_mode="bilinear")
        upsampled = F.interpolate(attrs, batch_x.shape[-2:], mode="bilinear", align_corners=True)
        # GradCAM aggregates over channels, check if we need to duplicate attributions in order to match input shape
        if upsampled.shape[1] != batch_x.shape[1]:
            upsampled = upsampled.repeat(1, batch_x.shape[1], 1, 1)
        return upsampled
