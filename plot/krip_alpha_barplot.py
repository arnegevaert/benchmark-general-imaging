import argparse
import os
import pandas as pd
from attrbench.lib import krippendorff_alpha
from attrbench.suite import SuiteResult
from experiments.general_imaging.plot.dfs import get_default_dfs, get_all_dfs
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from os import path
import glob
import seaborn as sns
from tqdm import tqdm


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("in_dir", type=str)
    parser.add_argument("out_file", type=str)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("-d", "--datasets", type=str, nargs="*")
    args = parser.parse_args()

    mpl.use("Agg")
    np.seterr(all="raise")
    sns.set()

    if args.datasets is None:
        result_files = glob.glob(path.join(args.in_dir, "*.h5"))
    else:
        result_files = [path.join(args.in_dir, f"{ds}.h5") for ds in args.datasets]
    result_objects = {path.basename(file).split(".")[0]: SuiteResult.load_hdf(file) for file in result_files}

    k_a = dict()
    for ds_name, res in tqdm(result_objects.items()):
        if args.all:
            dfs = get_all_dfs(res, mode="single").items()
        else:
            dfs = get_default_dfs(res, mode="single").items()
        k_a[ds_name] = {
            metric_name: krippendorff_alpha(df.to_numpy())
            for metric_name, (df, _) in dfs
        }
    k_a = pd.DataFrame(k_a)

    k_a = k_a.reindex(["mnist", "fashionmnist", "cifar10", "cifar100", "svhn", "imagenet", "caltech", "places"], axis=1)
    color_list = [
        "#036d08",
        "#9de78c",
        "#08036d",
        "#845eb3",
        "#e7c3ff",
        "#6d012a",
        "#b65a73",
        "#ffaac4"
    ]

    color_dict = {
        "mnist": "#036d08",
        "fashionmnist": "#9de78c",
        "cifar10": "#08036d",
        "cifar100": "#845eb3",
        "svhn": "#e7c3ff",
        "imagenet": "#6d012a",
        "caltech": "#b65a73",
        "places": "#ffaac4"
    }

    if args.all:
        fig, axs = plt.subplots(nrows=3, figsize=(10, 12), constrained_layout=True)

        for i in range(3):
            end = min((i+1)*14, k_a.shape[0])
            k_a[i*12:end].plot.bar(ax=axs[i], color=color_list, width=0.7)
            #plt.xticks(rotation=45, ha="right", rotation_mode="anchor")
            axs[i].set_xticklabels(axs[i].get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
            if i > 0:
                axs[i].legend().set_visible(False)
            else:
                axs[i].legend(loc='upper center', bbox_to_anchor=(0.5, 1.4),
                          fancybox=True, shadow=True, ncol=4)
            plt.grid(axis="x")
        fig.savefig(args.out_file, bbox_inches="tight", dpi=250)
    else:
        fig, ax = plt.subplots(figsize=(10, 4))
        k_a.plot.bar(ax=ax, color=color_list, width=0.7)
        plt.xticks(rotation=45, ha="right", rotation_mode="anchor")
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.25),
                      fancybox=True, shadow=True, ncol=4)
        plt.grid(axis="x")
        fig.savefig(args.out_file, bbox_inches="tight", dpi=250)
