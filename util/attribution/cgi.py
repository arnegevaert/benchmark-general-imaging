from attribench import AttributionMethod
from tqdm import trange
from captum import attr
from torch import nn
import torch


class CGI(AttributionMethod):
    def __init__(self, model: nn.Module):
        super().__init__(model)
        self.ixg = attr.InputXGradient(model)

    def __call__(
        self, batch_x: torch.Tensor, batch_target: torch.Tensor
    ) -> torch.Tensor:
        out = self.model(batch_x)
        num_logits = out.shape[1]
        print(num_logits)

        all_attrs = []
        for i in trange(num_logits):
            attrs = self.ixg.attribute(batch_x, target=i)
            all_attrs.append(attrs.detach().cpu())
        # (batch_size, num_logits, channels, height, width)
        all_attrs = torch.stack(all_attrs, dim=1)

        # (batch_size, channels, height, width)
        max_attrs = torch.max(all_attrs, dim=1, keepdim=False)[0]
        min_attrs = torch.min(all_attrs, dim=1, keepdim=False)[0]
        target_attrs = all_attrs[
            torch.arange(len(all_attrs)), batch_target.cpu()
        ]

        # Competition for pixels: only keep pixels that are the maximum absolute
        # attribution for their class
        target_attrs[target_attrs > 0 & target_attrs < max_attrs] = 0
        target_attrs[target_attrs < 0 & target_attrs > min_attrs] = 0

        return target_attrs
