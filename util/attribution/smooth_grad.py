from captum import attr
import torch
from torch import nn
from attribench import AttributionMethod


class SmoothGrad(AttributionMethod):
    def __init__(self, model: nn.Module, num_samples: int, stdev: float):
        self.method = attr.NoiseTunnel(attr.Saliency(model))
        self.num_samples = num_samples
        self.stdev = stdev

    def __call__(self, batch_x: torch.Tensor, batch_target: torch.Tensor):
        # follows paper more closely, but not perfectly.
        # in paper the sigma is set per image, here per batch
        # sigma = self.noise_level / (x.max()-x.min()) 
        return self.method.attribute(
            batch_x,
            target=batch_target,
            nt_type="smoothgrad",
            nt_samples=self.num_samples,
            nt_samples_batch_size=1,
            stdevs=self.stdev,
            abs=False
        )
