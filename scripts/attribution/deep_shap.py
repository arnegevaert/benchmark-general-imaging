from captum import attr
from torch.utils.data import Dataset, DataLoader
import torch
from torch import nn
from attrbench import AttributionMethod


class DeepShap(AttributionMethod):
    def __init__(self, model: nn.Module,
                 reference_dataset: Dataset,
                 n_baseline_samples: int):
        self.model = model
        self.reference_dataset = reference_dataset
        self.n_baseline_samples = n_baseline_samples
        self.method = attr.DeepLiftShap(model)
        self.ref_sampler = DataLoader(
            dataset=self.reference_dataset,
            batch_size=self.n_baseline_samples,
            shuffle=True, drop_last=True
        )

    def __call__(self, batch_x: torch.Tensor, batch_target: torch.Tensor):
        ref_batch = next(iter(self.ref_sampler))[0].to(batch_x.device)
        # Compute per sample to reduce VRAM usage
        return torch.cat([
            self.method.attribute(batch_x[i].unsqueeze(0),
                                  baselines=ref_batch,
                                  target=batch_target[i]).detach()
            for i in range(batch_x.shape[0])
        ])
