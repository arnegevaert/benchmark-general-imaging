from torch.utils.data import Dataset, DataLoader
from captum import attr
import torch
from torch import nn
from attrbench import AttributionMethod


class ExpectedGradients(AttributionMethod):
    def __init__(self, model: nn.Module,
                 reference_dataset: Dataset, num_samples: int):
        self.model = model
        self.reference_dataset = reference_dataset
        self.num_samples = num_samples
        self.method = attr.GradientShap(model)
        self.ref_sampler = DataLoader(
            dataset=self.reference_dataset,
            batch_size=self.num_samples,
            shuffle=True, drop_last=True
        )

    def __call__(self, batch_x: torch.Tensor, batch_target: torch.Tensor):
        ref_batch = next(iter(self.ref_sampler))[0].to(batch_x.device)
        # GradientSHAP has an option to add smoothing, but this is not done in original EG algorithm
        # Compute per sample to reduce VRAM usage
        return torch.cat([
            self.method.attribute(
                batch_x[i].unsqueeze(0),
                baselines=ref_batch, 
                target=batch_target[i],
                n_samples=self.num_samples, 
                stdevs=0.0).detach()
            for i in range(batch_x.shape[0])
        ])
