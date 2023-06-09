from attrbench import AttributionMethod
import torch
from torch import nn
import saliency.core as saliency
from torchray.attribution.rise import rise,rise_class

    
class Rise(AttributionMethod):
    def __init__(self, model: nn.Module,batch_size=16)-> None:
        super().__init__(model)
        self.batch_size=batch_size

    def __call__(self, batch_x: torch.Tensor, batch_target: torch.Tensor) -> torch.Tensor:
        channels=batch_x.shape[1]
        result=rise_class(self.model,input=batch_x,target=batch_target, batch_size=self.batch_size,resize=True)
        result=torch.tile(result,dims=[1,channels,1,1])
        return result
    
        