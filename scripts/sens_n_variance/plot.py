import argparse
import pandas as pd
import matplotlib as mpl
import os
import seaborn as sns
import matplotlib.pyplot as plt


def snr(df):
    mu = df.groupby("Sample").mean().melt()["value"]
    sigma = df.groupby("Sample").std().melt()["value"]
    return ((mu**2) / (sigma**2)).to_numpy()


def frac_var(df):
    total_var = df.melt().groupby("variable").var()
    sample_var = pd.melt(df.reset_index(), id_vars=["Sample"]).groupby(
         ["Sample", "variable"]).var()
    return ((sample_var / total_var) * 100).to_numpy().flatten()


def make_snr_dataframe(results, ds_name, metric_name):
    snr_values = snr(results)
    return pd.DataFrame(
        {
            "SNR": snr_values,
            "Dataset": [ds_name] * snr_values.shape[0],
            "Metric": [metric_name] * snr_values.shape[0],
        }
    )


def make_frac_var_dataframe(results, ds_name, metric_name):
    frac_var_values = frac_var(results)
    return pd.DataFrame(
        {
            "Noise % var": frac_var_values,
            "Dataset": [ds_name] * frac_var_values.shape[0],
            "Metric": [metric_name] * frac_var_values.shape[0],
        }
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--in-dir", type=str)
    parser.add_argument("-o", "--out-dir", type=str)
    args = parser.parse_args()

    mpl.use("Agg")
    sns.set()
    
    # Check if output directory exists and is empty
    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)
    elif len(os.listdir(args.out_dir)) > 0:
        raise ValueError("Output directory is not empty")

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
            sens_n_result = pd.read_csv(
                os.path.join(ds_path, "sens_n.csv"),
                delimiter=",",
                index_col="Sample",
            )
            seg_sens_n_result = pd.read_csv(
                os.path.join(ds_path, "seg_sens_n.csv"),
                delimiter=",",
                index_col="Sample",
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
