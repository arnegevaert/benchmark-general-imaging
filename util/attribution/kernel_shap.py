from captum import attr
from skimage.segmentation import slic
import numpy as np
import torch
from attribench import AttributionMethod
from torch import nn


class KernelShap(AttributionMethod):
    def __init__(
        self,
        model: nn.Module,
        num_samples: int,
        super_pixels = False,
        num_segments: int | None = None,
    ):
        if super_pixels and num_segments is None:
            raise ValueError(
                f"num_segments cannot be None when using super_pixels"
            )
        self.num_segments = num_segments
        self.super_pixels = super_pixels
        self.method = attr.KernelShap(model)
        self.num_samples = num_samples

    def __call__(self, batch_x: torch.Tensor, batch_target: torch.Tensor):
        internal_batch_size = batch_x.shape[0]
        
        if self.super_pixels:
            masks = get_super_pixels(batch_x, self.num_segments)
            feature_masks = [masks[i, ...].unsqueeze(0) for i in range(batch_x.shape[0])]
        else:
            feature_masks = [None for _ in range(batch_x.shape[0])]

        # Compute KernelSHAP this per-sample to avoid warnings.
        # If we run KernelSHAP on multiple samples,
        # captum runs it per sample in a for loop anyway.
        # With the perturbations_per_eval parameter, we can control how many
        # perturbed samples are run in parallel.
        return torch.cat(
            [
                self.method.attribute(
                    batch_x[i, ...].unsqueeze(0),
                    target=batch_target[i],
                    feature_mask=feature_masks[i],
                    n_samples=self.num_samples,
                    perturbations_per_eval=internal_batch_size,
                )
                for i in range(batch_x.shape[0])
            ],
            dim=0,
        )


def get_super_pixels(x, k):
    images = x.detach().cpu().numpy()
    # assuming grayscale images have 1 channel
    nr_of_channels = images.shape[1]
    masks = []
    for i in range(images.shape[0]):
        input_image = np.transpose(images[i], (1, 2, 0))
        mask = slic(input_image, n_segments=k, slic_zero=True, start_label=0)
        masks.append(mask)
    masks = torch.LongTensor(np.stack(masks))
    masks = masks.unsqueeze(dim=1)
    return masks.expand(-1, nr_of_channels, -1, -1).to(x.device)
