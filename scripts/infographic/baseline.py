from experiments.general_imaging.lib.dataset_models import get_dataset_model
from experiments.lib.attribution.edge_detection import EdgeDetection
import matplotlib.pyplot as plt
import torch
import matplotlib as mpl


"""
This script is used to generate baseline heatmaps for use in the infographic in the paper.
"""
if __name__ == "__main__":
    ds, model, patch_dir = get_dataset_model("ImageNet", model_name="resnet18")
    idx = 2011

    # Get original image
    img = ds[idx][0]
    mean = torch.tensor([.485, .456, .406]).reshape(3, 1, 1)
    std = torch.tensor([.229, .224, .225]).reshape(3, 1, 1)
    img = img * std + mean
    fig, ax = plt.subplots()
    ax.imshow(img.permute(1, 2, 0))
    plt.axis("off")
    fig.savefig("img.png", bbox_inches="tight")
    plt.close(fig)

    # Generate random heatmap
    rand_res = torch.rand(img.shape).mean(dim=0)
    fig, ax = plt.subplots()
    ax.imshow(rand_res, cmap="Reds")
    plt.axis("off")
    fig.savefig("rand.png", bbox_inches="tight")
    plt.close(fig)

    # Generate Sobel edge detection heatmap
    ed = EdgeDetection()
    ed_res = ed(img, target=0).mean(dim=0)
    fig, ax = plt.subplots()
    ax.imshow(ed_res, cmap="Reds")
    plt.axis("off")
    fig.savefig("ed.png", bbox_inches="tight")
    plt.close(fig)
