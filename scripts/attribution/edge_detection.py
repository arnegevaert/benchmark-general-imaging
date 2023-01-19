import torch
from skimage.filters import sobel


class EdgeDetection:
    def __call__(self, x, target):
        device = x.device
        x = x.detach().cpu().numpy()
        x = (x - x.min()) / (x.max() - x.min())
        for i in range(x.shape[0]):
            for channel in range(x.shape[1]):
                x[i, channel] = sobel(x[i, channel])
        attrs = torch.tensor(x).to(device)
        return attrs