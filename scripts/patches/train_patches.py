import argparse
import torch
from lib import get_dataset_model
from attrbench.lib import make_patch
from torch.utils.data import DataLoader
from os import path
import os


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset", type=str, required=True)
    parser.add_argument("-m", "--model", type=str, required=True)
    parser.add_argument("-b", "--batch-size", type=int, required=True)
    parser.add_argument("-c", "--cuda", action="store_true")
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--epochs", type=int, default=5)
    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() and args.cuda else "cpu"

    dataset, model, patch_folder = get_dataset_model(args.dataset, model_name=args.model)
    model.to(device)
    model.eval()

    if not path.isdir(patch_folder):
        os.makedirs(patch_folder)

    dl = DataLoader(dataset, batch_size=args.batch_size, num_workers=4)
    for target in range(10):
        patch_name = f"patch_{target}.pt"
        print(f"{patch_name}...")
        make_patch(dl, model, target, path.join(patch_folder, patch_name),
                   device, epochs=args.epochs,
                   lr=args.lr, patch_percent=0.1)
