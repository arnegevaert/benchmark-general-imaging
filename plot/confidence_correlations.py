from attrbench.suite import SuiteResult
import os
import argparse
from tqdm import tqdm
from experiments.general_imaging.plot.dfs import get_default_dfs, get_all_dfs
from os import path
import numpy as np
from scipy.special import softmax
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("in_dir", type=str)
    parser.add_argument("out_dir", type=str)
    parser.add_argument("--all", action="store_true")
    args = parser.parse_args()

    params = [
        ("MNIST", "mnist"),
        ("FashionMNIST", "fashionmnist"),
        ("CIFAR10", "cifar10"),
        ("CIFAR100", "cifar100"),
        ("SVHN", "svhn"),
        ("Places365", "places"),
        ("Caltech256", "caltech"),
        ("ImageNet", "imagenet")
    ]
    method = "spearman"
    use_softmax = False

    mpl.use("Agg")

    if not path.isdir(args.out_dir):
        os.makedirs(args.out_dir)

    prog = tqdm(params)
    for ds_name, filename in prog:
        hdf_name = filename + ".h5"
        csv_name = filename + ".csv"
        res_obj = SuiteResult.load_hdf(path.join(args.in_dir, hdf_name))
        prog.set_postfix_str(filename)

        for mode in "single", "raw":
            if args.all:
                dfs = get_all_dfs(res_obj, mode=mode)
            else:
                dfs = get_default_dfs(res_obj, mode=mode)
            logits = np.loadtxt(path.join(args.in_dir, "confidence", csv_name), delimiter=",")
            if use_softmax:
                confidences = np.max(softmax(logits, axis=1), axis=1)
            else:
                confidences = np.max(logits, axis=1)

            corrs = {}
            for key, (df, inverted) in dfs.items():
                df = -df if inverted else df
                if key in ("deletion_morf", "deletion_lerf", "insertion_morf", "insertion_lerf"):
                    df = df.div(pd.Series(confidences), axis=0)
                corrs[key] = df.corrwith(pd.Series(confidences), method=method)
            df = pd.DataFrame(corrs)

            figsize = (15, 15) if not args.all else (100, 15)
            fig, ax = plt.subplots(figsize=figsize)
            sns.heatmap(df, annot=True, vmin=-1, vmax=1, cmap=sns.diverging_palette(220, 20, as_cmap=True), ax=ax)
            ax.set_title(ds_name)
            ax.set_aspect("equal")
            fig.savefig(path.join(args.out_dir, f"{filename}_{mode}.png"), bbox_inches="tight")
            plt.close(fig)
