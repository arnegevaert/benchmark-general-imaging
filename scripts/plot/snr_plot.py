import argparse
import pandas as pd
import numpy as np
import matplotlib as mpl
import os
import seaborn as sns
import matplotlib.pyplot as plt


def snr(arr):
    mu = np.mean(arr, axis=1)
    sigma = np.std(arr, axis=1)
    return (mu**2) / (sigma**2)


def frac_var(arr):
    total_var = np.var(arr)
    sample_var = np.var(arr, axis=1)
    return (sample_var / total_var) * 100


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--in-dir", type=str, default="out")
    parser.add_argument("-o", "--out-dir", type=str, default="plot")
    args = parser.parse_args()

    mpl.use("Agg")
    sns.set()

    datasets = [
        "MNIST",
        "FashionMNIST",
        "CIFAR10",
        "CIFAR100",
        "SVHN",
        "ImageNet",
        "Caltech",
        "Places",
    ]

    snr_dfs = []
    var_dfs = []
    for ds_name in datasets:
        ds_path = os.path.join(args.in_dir, ds_name)
        pass #TODO