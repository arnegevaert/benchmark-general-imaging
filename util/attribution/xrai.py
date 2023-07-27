
from attrbench import AttributionMethod
import torch
import saliency.core as saliency
import numpy as np

class XRAI(AttributionMethod):
    def __init__(self, model: torch.nn.Module, batch_size: int,**kwargs) -> None:
        super().__init__(model)
        self.batch_size = batch_size
        self.xrai_object = saliency.XRAI()

    
    def _call_model_function_xrai(self,images, call_model_args=None, expected_keys=None):
        with torch.enable_grad():
            class_idx_str = 'class_idx_str'
            device = call_model_args['device']
            images=images.transpose(0,3,1,2)
            images=torch.tensor(images, dtype=torch.float32, device=device)
            images= images.requires_grad_(True)
            target_class_idx =  call_model_args[class_idx_str]
            output = self.model(images)
            # m = torch.nn.Softmax(dim=1)
            # output = m(output)
            assert saliency.base.INPUT_OUTPUT_GRADIENTS in expected_keys
            outputs = output[:,target_class_idx]
            grads = torch.autograd.grad(outputs, images, grad_outputs=torch.ones_like(outputs))
            # grads = torch.movedim(grads[0], 1, 3)
            gradients = grads[0].detach().cpu().numpy()
            gradients = gradients.transpose(0,2,3,1)
            return {saliency.base.INPUT_OUTPUT_GRADIENTS: gradients}

    def __call__(self, batch_x: torch.Tensor, batch_target: torch.Tensor) -> torch.Tensor:
        class_idx_str = 'class_idx_str'
        device = batch_x.device
        batch_x=batch_x.detach().cpu().numpy()
        channels=batch_x.shape[1]
        batch_target=batch_target.detach().cpu().numpy()
        res_list= []
        for im, target in zip(batch_x,batch_target):
            call_model_args = {class_idx_str: target, 'device':device}
            xrai_attributions = self.xrai_object.GetMask(im.transpose(1,2,0), self._call_model_function_xrai, call_model_args, batch_size=self.batch_size)
            res_list.append(xrai_attributions)
        result = np.stack(res_list)
        # expand attribtions to have channels for compatibility reasons
        result=np.tile(np.expand_dims(result,axis=1),(1,channels,1,1))
        result=torch.from_numpy(result)
        return result
    
        