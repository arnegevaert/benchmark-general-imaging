import argparse
from typing import Dict
import torch
import os
from util.models import get_model
from util.datasets import ALL_DATASETS
from attribench.data import HDF5Dataset
from attribench.functional import compute_attributions
from util.attribution import GradCAMPP, ScoreCAM


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=str)
    parser.add_argument("--data-dir", type=str)
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()

    attributions = {}
    for ds_name in ["MNIST", "FashionMNIST", "SVHN", "CIFAR10", "CIFAR100"]:
        model_name = (
            "BasicCNN" if ds_name in ["MNIST", "FashionMNIST"] else "resnet20"
        )
        dataset = HDF5Dataset(
            os.path.join(args.out_dir, ds_name, "samples.h5")
        )
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = get_model(ds_name, args.data_dir, model_name)
        method_dict = {
            "GradCAM++": GradCAMPP(model),
            "ScoreCAM": ScoreCAM(model),
        }

        attributions[ds_name] = compute_attributions(
            model,
            method_dict,
            dataset,
            batch_size=args.batch_size,
            device=device,
        )

        assert isinstance(attributions[ds_name], Dict)
        any_nans = False
        for key in attributions[ds_name].keys():
            has_nans = torch.isnan(attributions[ds_name][key]).any().item()
            any_nans = has_nans or any_nans
            if has_nans:
                print(f"{ds_name}: Found NaNs in {key}!")
        if not any_nans:
            print(f"{ds_name}: No NaNs found!")