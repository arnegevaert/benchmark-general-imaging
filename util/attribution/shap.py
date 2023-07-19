from captum import attr
from captum._utils.models.linear_model import SkLearnLinearModel
from skimage import segmentation
import numpy as np
import torch
from torch.utils.data import DataLoader
from attribench import AttributionMethod
from torch import nn


class TabularShap(AttributionMethod):
    def __init__(
        self, model: nn.Module, n_samples, baselines=None, feature_mask=None
    ):
        self.method = attr.ShapleyValueSampling(model)
        self.n_samples = n_samples
        if isinstance(baselines, list):
            baselines = torch.Tensor(baselines)[None]
        self.baselines = baselines
        self.feature_mask = feature_mask

    def __call__(self, batch_x: torch.Tensor, batch_target: torch.Tensor):
        return self.method.attribute(
            batch_x,
            baselines=self.baselines,
            target=batch_target,
            feature_mask=self.feature_mask,
        )


class TabularLime:
    def __init__(self, model, n_samples, baselines=None, feature_mask=None):
        self.method = attr.Lime(
            model,
            interpretable_model=SkLearnLinearModel(
                "linear_model.LinearRegression"
            ),
        )  # defaolt lasso model always learns to output 0.
        self.n_samples = n_samples
        if isinstance(baselines, list):
            baselines = torch.Tensor(baselines)[None]
        self.baselines = baselines
        self.feature_mask = feature_mask

    def __call__(self, x, target):
        return self.method.attribute(
            x,
            baselines=self.baselines,
            target=target,
            feature_mask=self.feature_mask,
        )


class KernelShap(AttributionMethod):
    def __init__(
        self,
        model: nn.Module,
        num_samples: int,
        super_pixels=True,
        num_segments: int | None = None,
        feature_mask=None,
        baselines=None,
    ):
        if super_pixels and num_samples is None:
            raise ValueError(
                f"n_segments cannot be None when using super_pixels"
            )
        self.num_segments = num_segments
        self.super_pixels = super_pixels
        self.method = attr.KernelShap(model)
        self.num_samples = num_samples
        self.feature_mask = feature_mask
        if isinstance(baselines, list):
            baselines = torch.Tensor(baselines)[None]
        self.baselines = baselines

    def __call__(self, batch_x: torch.Tensor, batch_target: torch.Tensor):
        if self.feature_mask is not None:
            masks = self.feature_mask
        else:
            masks = (
                get_super_pixels(batch_x, self.num_segments)
                if self.super_pixels
                else None
            )
        # Compute KernelSHAP this per-sample to avoid warnings.
        # If we run KernelSHAP on multiple samples,
        # captum runs it per sample in a for loop anyway.
        # With the perturbations_per_eval parameter, we can control how many
        # perturbed samples are run in parallel.
        internal_batch_size = x.shape[0]
        return torch.cat(
            [
                self.method.attribute(
                    batch_x[i, ...].unsqueeze(0),
                    target=batch_target[i],
                    feature_mask=masks[i, ...].unsqueeze(0),
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
        mask = segmentation.slic(
            input_image, n_segments=k, slic_zero=True, start_label=0
        )
        masks.append(mask)
    masks = torch.LongTensor(np.stack(masks))
    masks = masks.unsqueeze(dim=1)
    return masks.expand(-1, nr_of_channels, -1, -1).to(x.device)
