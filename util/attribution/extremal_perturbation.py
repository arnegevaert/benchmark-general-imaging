from attrbench import AttributionMethod
import torch
from torch import nn
import saliency.core as saliency
from torchray.attribution.extremal_perturbation import extremal_perturbation, contrastive_reward, simple_reward

class ExtremalPerturbation(AttributionMethod):
    def __init__(self, model: nn.Module,reward = 'contrastive_reward', areas=[0.1])-> None:
        super().__init__(model)
        if reward=='contrastive_reward':
            self.reward_fuction = contrastive_reward
        else:
            self.reward_fuction = simple_reward
        self.areas=areas

    def __call__(self, batch_x: torch.Tensor, batch_target: torch.Tensor) -> torch.Tensor:
        attributions=[]
        channels=batch_x.shape[1]
        for x,y in zip(batch_x,batch_target):
            with torch.enable_grad():
                res=extremal_perturbation(self.model,x.unsqueeze(0),int(y),areas=self.areas, reward_func=self.reward_fuction,resize=True)
            mask = res[0]
            attributions.append(mask.detach().cpu())

        result=torch.vstack(attributions)
        result=torch.tile(result,dims=[1,channels,1,1])
        
        return result
        