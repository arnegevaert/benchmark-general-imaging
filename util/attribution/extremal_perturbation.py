from attrbench import AttributionMethod
import torch
from torch import nn
import saliency.core as saliency
from .extremal.extremal_perturbation import extremal_perturbation, contrastive_reward, simple_reward

class ExtremalPerturbation_batched(AttributionMethod):
    def __init__(self, model: nn.Module,reward = 'simple_reward', areas=0.1)-> None:
        super().__init__(model)
        if reward=='simple_reward':
            self.reward_fuction = simple_reward
        else:
            self.reward_fuction = contrastive_reward
        self.areas=areas

    def __call__(self, batch_x: torch.Tensor, batch_target: torch.Tensor) -> torch.Tensor:

        channels=batch_x.shape[1]
        with torch.enable_grad():
            res=extremal_perturbation(self.model,batch_x,batch_target,areas=self.areas, reward_func=self.reward_fuction,resize=True)
        result = res[0].detach().cpu()

        result=torch.tile(result,dims=[1,channels,1,1])
        
        return result