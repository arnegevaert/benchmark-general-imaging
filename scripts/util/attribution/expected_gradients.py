from torch.utils.data import Dataset, DataLoader
from captum import attr
import torch


class ExpectedGradients:
    def __init__(self, model, reference_dataset: Dataset, num_samples: int):
        self.model = model
        self.reference_dataset = reference_dataset
        self.num_samples = num_samples
        self.method = attr.GradientShap(model)
        self.ref_sampler = DataLoader(
            dataset=self.reference_dataset,
            batch_size=self.num_samples,
            shuffle=True, drop_last=True
        )

    def __call__(self, x, target):
        ref_batch = next(iter(self.ref_sampler))[0].to(x.device)
        # GradientSHAP has an option to add smoothing, but this is not done in original EG algorithm
        # Compute per sample to reduce VRAM usage
        return torch.cat([
            self.method.attribute(
                x[i].unsqueeze(0),
                baselines=ref_batch, 
                target=target[i],
                n_samples=self.num_samples, 
                stdevs=0.0).detach()
            for i in range(x.shape[0])
        ])
