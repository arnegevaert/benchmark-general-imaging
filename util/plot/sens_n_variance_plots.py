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
    parser.add_argument("in_dir", type=str)
    parser.add_argument("out_file", type=str)
    args = parser.parse_args()

    mpl.use("Agg")
    sns.set()

    datasets = [
        "mnist",
        "fashionmnist",
        "cifar10",
        "cifar100",
        "svhn",
        "imagenet",
        "caltech",
        "places",
    ]
    snr_dfs = []
    var_dfs = []
    for ds_name in datasets:
        ds_path = os.path.join(args.in_dir, ds_name)
        sens_results = np.loadtxt(
            os.path.join(ds_path, "sens_n.csv"), delimiter=","
        )
        seg_results = np.loadtxt(
            os.path.join(ds_path, "seg_sens_n.csv"), delimiter=","
        )
        snr_dfs.append(
            pd.DataFrame(
                {
                    "SNR": snr(sens_results),
                    "Dataset": [ds_name] * sens_results.shape[0],
                    "Metric": ["Sensitivity-n"] * sens_results.shape[0],
                }
            )
        )
        snr_dfs.append(
            pd.DataFrame(
                {
                    "SNR": snr(seg_results),
                    "Dataset": [ds_name] * seg_results.shape[0],
                    "Metric": ["Seg-Sensitivity-n"] * seg_results.shape[0],
                }
            )
        )

        var_dfs.append(
            pd.DataFrame(
                {
                    "Noise % var": frac_var(sens_results),
                    "Dataset": [ds_name] * sens_results.shape[0],
                    "Metric": ["Sensitivity-n"] * sens_results.shape[0],
                }
            )
        )
        var_dfs.append(
            pd.DataFrame(
                {
                    "Noise % var": frac_var(seg_results),
                    "Dataset": [ds_name] * seg_results.shape[0],
                    "Metric": ["Seg-Sensitivity-n"] * seg_results.shape[0],
                }
            )
        )

    snr_df = pd.concat(snr_dfs)
    var_df = pd.concat(var_dfs)

    fig, axs = plt.subplots(1, 2, figsize=(15, 5))
    p = sns.boxplot(x="Dataset", hue="Metric", y="SNR", data=snr_df, ax=axs[0])
    p.set_yscale("log")
    p.set_xticklabels(p.get_xticklabels(), rotation=30)

    p = sns.barplot(
        x="Dataset", hue="Metric", y="Noise % var", data=var_df, ax=axs[1]
    )
    p.set_xticklabels(p.get_xticklabels(), rotation=30)
    # fig.savefig(args.out_file, bbox_inches="tight", dpi=250)
    fig.savefig(args.out_file, bbox_inches="tight")
