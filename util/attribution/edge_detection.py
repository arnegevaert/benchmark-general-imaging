import torch
from skimage.filters import sobel
import torch
from attribench import AttributionMethod


class EdgeDetection(AttributionMethod):
    def __call__(self, x: torch.Tensor, batch_target: torch.Tensor):
        device = x.device
        x = x.detach().cpu().numpy()
        x = (x - x.min()) / (x.max() - x.min())
        for i in range(x.shape[0]):
            for channel in range(x.shape[1]):
                x[i, channel] = sobel(x[i, channel])
        attrs = torch.tensor(x).to(device)
        return attrs
