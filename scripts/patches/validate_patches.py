import argparse
import torch
import numpy as np
from torch.utils.data import DataLoader
from os import path, listdir
import os
import re
from tqdm import tqdm
from util.models import get_model
from util.datasets import get_dataset


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument("--patch-dir", type=str, required=True)
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    dataset = get_dataset(args.dataset, args.data_dir)
    model = get_model(args.dataset, args.data_dir, args.model)
    model.to(device)
    model.eval()

    dl = DataLoader(dataset, batch_size=args.batch_size, num_workers=4)
    patch_names = [
        filename
        for filename in listdir(args.patch_dir)
        if filename.endswith(".pt")
    ]
    if len(patch_names) == 0:
        raise ValueError("No patches found in patch directory")

    for name in patch_names:
        preds = []
        target_expr = re.compile(r".*_([0-9]*)\.pt")
        target = int(target_expr.match(name).group(1))
        for samples, labels in tqdm(dl):
            patch = torch.load(
                path.join(args.patch_dir, name),
                map_location=lambda storage, loc: storage,
            ).to(device)
            atk_samples = samples.clone().to(device)
            indx = samples.shape[-1] // 2 - patch.shape[-1] // 2
            indy = indx
            atk_samples[
                :,
                :,
                indx : indx + patch.shape[-1],
                indy : indy + patch.shape[-1],
            ] = patch
            atk_out = model(atk_samples)
            preds.append(atk_out.argmax(axis=1).detach().cpu().numpy())
        preds = np.concatenate(preds)
        print(
            f"{name}: Fraction of successful attacks:",
            np.count_nonzero(preds == target) / preds.shape[0],
        )
