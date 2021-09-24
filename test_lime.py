from experiments.lib.attribution import ImageLime
from experiments.general_imaging.lib.dataset_models import get_dataset_model
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
import torch

ds, model, _ = get_dataset_model("ImageNet", model_name="resnet18")
model.to("cuda")
model.eval()
dl = DataLoader(ds, batch_size=4)
batch, labels = next(iter(dl))
batch = batch.to("cuda")
labels = labels.to("cuda")
batch = (batch - batch.min()) / (batch.max() - batch.min())

lime = ImageLime(model)
exp = lime(batch, labels).float()
exp = (exp - exp.min()) / (exp.max() - exp.min())
plt.imshow(make_grid(torch.cat([batch, exp], dim=0)).permute(1, 2, 0).cpu().numpy())
