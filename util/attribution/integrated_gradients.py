from captum import attr
import torch
from torch import nn
from attribench import AttributionMethod


class IntegratedGradients(AttributionMethod):
    def __init__(
        self, model: nn.Module, internal_batch_size: int, num_steps: int
    ):
        self.method = attr.IntegratedGradients(model)
        self.internal_batch_size = internal_batch_size
        self.num_steps = num_steps

    def __call__(self, batch_x: torch.Tensor, batch_target: torch.Tensor):
        return self.method.attribute(
            batch_x,
            target=batch_target,
            internal_batch_size=self.internal_batch_size,
            n_steps=self.num_steps,
        )
