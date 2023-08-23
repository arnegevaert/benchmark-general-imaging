from util.torchcam.methods.activation import ScoreCAM as TCScoreCAM
from .gradcam import GradCAM
import torch


class ScoreCAM(GradCAM):
    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__(model)

    def __call__(
        self, batch_x: torch.Tensor, batch_target: torch.Tensor
    ) -> torch.Tensor:
        scorecam = TCScoreCAM(
            self.model, target_layer=self.model.get_last_conv_layer()
        )
        out = self.model(batch_x)
        result_list = scorecam(class_idx=batch_target.tolist(), scores=out)
        scorecam.remove_hooks()

        result = result_list[0].unsqueeze(1)
        # Shape of result is now (batch_size, 1, height, width)
        # Reshape attributions to match input shape
        return self._reshape_attrs(result, batch_x)
