import argparse
import torch
from experiments.general_imaging.lib.dataset_models import get_dataset_model
from attrbench.suite import SuiteResult
from os import path
import numpy as np
import os


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("out_dir", type=str)
    parser.add_argument("-c", "--cuda", action="store_true")
    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() and args.cuda else "cpu"

    if not path.isdir(args.out_dir):
        os.makedirs(args.out_dir)

    hdf_loc = "../../out"
    params = [
        ("MNIST", "mnist.h5", "CNN"),
        ("FashionMNIST", "fashionmnist.h5", "CNN"),
        ("CIFAR10", "cifar10.h5", "resnet20"),
        ("CIFAR100", "cifar100.h5", "resnet20"),
        ("SVHN", "svhn.h5", "resnet20"),
        ("ImageNet", "imagenet.h5", "resnet18"),
        ("Places365", "places.h5", "resnet18"),
        ("Caltech256", "caltech.h5", "resnet18"),
    ]

    for ds_name, hdf_name, model_name in params:
        print(ds_name)
        ds, model, _ = get_dataset_model(ds_name, model_name=model_name)
        model.eval()
        model.to(device)
        res_obj = SuiteResult.load_hdf(path.join(hdf_loc, hdf_name))
        all_imgs = res_obj.images
        outputs = []
        csv_name = hdf_name.split('.')[0] + ".csv"
        for i in range(4):
            batch = torch.tensor(all_imgs[i*64:(i+1)*64]).float().to(device)
            outputs.append(model(batch).detach().cpu().numpy())
        outputs = np.concatenate(outputs, axis=0)
        np.savetxt(path.join(args.out_dir, csv_name), outputs, delimiter=",")
