import types
import torch
from torch import nn
from torchvision.transforms import GaussianBlur
from attribench import AttributionMethod
from .igos_lib.methods import IGOS, iGOS_p, iGOS_pp
from .igos_lib.methods_helper import *
from typing import Callable

args = types.SimpleNamespace(size=28,
                             batch_size=1,
                             L1=1,
                             L2=20,
                             ig_iter=20,
                             iterations=15,
                             alpha=1000,
                             )

class _Igos_base(AttributionMethod):
    def __init__(self, model: nn.Module, **kwargs) -> None:
        super().__init__(model)
        self.blur = GaussianBlur((51, 51), sigma=50)
        self.method: Callable

    def __call__(self, batch_x: torch.Tensor, batch_target: torch.Tensor) -> torch.Tensor:
       channels=batch_x.shape[1]
       init(batch_x.shape[-1])
       device = batch_x.device
       blurs = self.blur(batch_x)
       with torch.enable_grad():
           masks = self.method(self.model,
                images=batch_x.detach(),
                baselines=blurs.detach(),
                labels=batch_target,
                size=args.size,
                iterations=args.ig_iter,
                ig_iter=args.iterations,
                L1=args.L1,
                L2=args.L2,
                alpha=args.alpha,
        )
       attrs = upscale(masks)
       attrs=torch.tile(attrs,(1,channels,1,1))
       return attrs


class Igos(_Igos_base):
    def __init__(self, model: nn.Module, **kwargs) -> None:
        super().__init__(model, **kwargs)
        self.method=IGOS

class Igos_pp(_Igos_base):
    def __init__(self, model: nn.Module, **kwargs) -> None:
        super().__init__(model, **kwargs)
        self.method=iGOS_pp