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


def make_snr_dataframe(results, ds_name, metric_name):
    return pd.DataFrame(
        {
            "SNR": snr(results),
            "Dataset": [ds_name] * results.shape[0],
            "Metric": [metric_name] * results.shape[0],
        }
    )


def make_frac_var_dataframe(results, ds_name, metric_name):
    return pd.DataFrame(
        {
            "Noise % var": frac_var(results),
            "Dataset": [ds_name] * results.shape[0],
            "Metric": [metric_name] * results.shape[0],
        }
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i", "--in-dir", type=str, default="out/results/sens_n_variance"
    )
    parser.add_argument(
        "-o", "--out-dir", type=str, default="out/plots/sens_n_variance"
    )
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
        if os.path.isdir(ds_path):
            sens_n_result = np.loadtxt(
                os.path.join(ds_path, "sens_n.csv"), delimiter=","
            )
            seg_sens_n_result = np.loadtxt(
                os.path.join(ds_path, "seg_sens_n.csv"), delimiter=","
            )

            snr_dfs.append(
                make_snr_dataframe(sens_n_result, ds_name, "Sensitivity-n")
            )
            snr_dfs.append(
                make_snr_dataframe(
                    seg_sens_n_result, ds_name, "Seg-Sensitivity-n"
                )
            )

            var_dfs.append(
                make_frac_var_dataframe(
                    sens_n_result, ds_name, "Sensitivity-n"
                )
            )
            var_dfs.append(
                make_frac_var_dataframe(
                    seg_sens_n_result, ds_name, "Seg-Sensitivity-n"
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
    fig.savefig(
        os.path.join(args.out_dir, "sens_n_variance.svg"), bbox_inches="tight"
    )
