from captum import attr
import torch
import numpy as np
from skimage import segmentation
from attribench import AttributionMethod
from torch import nn


class ImageLime(AttributionMethod):
    def __init__(self, model: nn.Module, num_samples, num_segments):
        super().__init__(model)
        # Cosine distance with kernel width=0.25 is the default for images
        # in the original LIME repository
        sim_fn = attr._core.lime.get_exp_kernel_similarity_function(
            kernel_width=0.25,
            distance_mode="cosine"
        )
        self.method = attr.Lime(self.model, similarity_func=sim_fn)
        self.num_samples = num_samples
        self.num_segments = num_segments

    def __call__(self, batch_x: torch.Tensor, batch_target: torch.Tensor):
        # Segment the images using SLIC
        images = batch_x.detach().cpu().numpy()
        num_channels = images.shape[1]
        masks = []
        for i in range(images.shape[0]):
            img = np.transpose(images[i], (1, 2, 0))
            mask = segmentation.slic(
                img, start_label=0, n_segments=self.num_segments
            )
            masks.append(mask)
        masks = np.array(masks)
        masks = torch.tensor(
            data=masks, device=batch_x.device, dtype=torch.long
        )
        masks = masks.unsqueeze(dim=1).expand(-1, num_channels, -1, -1)

        # Next, compute LIME. Do this per-sample to avoid warnings.
        # If we run LIME on multiple samples,
        # captum runs it per sample in a for loop anyway.
        # With the perturbations_per_eval parameter, we can control how many
        # perturbed samples are run in parallel.
        internal_batch_size = batch_x.shape[0]
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
