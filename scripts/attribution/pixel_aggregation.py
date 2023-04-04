import torch
from typing import Callable


class PixelAggregation:
    def __init__(self, base_method: Callable, aggregation_fn: str):
        self.base_method = base_method
        if aggregation_fn not in ["max_abs", "avg"]:
            raise ValueError("aggregation_fn must be max_abs or avg")
        self.aggregation_fn = aggregation_fn
    
    def __call__(self, x, target):
        attrs = self.base_method(x, target)
        if self.aggregation_fn == "max_abs":
            abs_value = attrs.abs()
            index = torch.argmax(abs_value, dim=1, keepdim=True)
            return torch.gather(attrs, dim=1, index=index)
        elif self.aggregation_fn == "avg":
            return torch.mean(attrs, dim=1, keepdim=True)