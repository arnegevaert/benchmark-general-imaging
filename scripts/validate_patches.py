import argparse
import torch
import numpy as np
from experiments.general_imaging.dataset_models import get_dataset_model
from attrbench.lib import validate
from torch.utils.data import DataLoader
from os import path, listdir
import os
import re
from tqdm import tqdm


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset", type=str, required=True)
    parser.add_argument("-m", "--model", type=str, required=True)
    parser.add_argument("-b", "--batch-size", type=int, required=True)
    parser.add_argument("-c", "--cuda", action="store_true")
    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() and args.cuda else "cpu"

    dataset, model, patch_folder = get_dataset_model(args.dataset, model_name=args.model)
    model.to(device)
    model.eval()

    if not path.isdir(patch_folder):
        os.makedirs(patch_folder)

    dl = DataLoader(dataset, batch_size=args.batch_size, num_workers=4)
    patch_names = [filename for filename in listdir(patch_folder) if filename.endswith(".pt")]
    for name in patch_names:
        preds = []
        target_expr = re.compile(r".*_([0-9]*)\.pt")
        target = int(target_expr.match(name).group(1))
        for samples, labels in tqdm(dl):
            patch = torch.load(path.join(patch_folder, name), map_location=lambda storage, loc: storage).to(device)
            atk_samples = samples.clone().to(device)
            indx = samples.shape[-1] // 2 - patch.shape[-1] // 2
            indy = indx
            atk_samples[:, :, indx:indx + patch.shape[-1], indy:indy + patch.shape[-1]] = patch
            atk_out = model(atk_samples)
            preds.append(atk_out.argmax(axis=1).detach().cpu().numpy())
        preds = np.concatenate(preds)
        print("Fraction of successful attacks:", np.count_nonzero(preds == target) / preds.shape[0])
