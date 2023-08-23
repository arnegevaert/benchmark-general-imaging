from util.torchcam.methods.gradient import GradCAMpp as TCGradCAMpp
import torch.nn.functional as F
import torch
from .gradcam import GradCAM


class GradCAMPP(GradCAM):
    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__(model)

    def __call__(
        self, batch_x: torch.Tensor, batch_target: torch.Tensor
    ) -> torch.Tensor:
        gradcampp = TCGradCAMpp(
            self.model, target_layer=self.model.get_last_conv_layer()
        )
        out = self.model(batch_x)
        result_list = gradcampp(class_idx=batch_target.tolist(), scores=out)
        gradcampp.remove_hooks()
        
        result = result_list[0].unsqueeze(1)
        # Shape of result is now (batch_size, 1, height, width)
        # Reshape attributions to match input shape
        return self._reshape_attrs(result, batch_x)