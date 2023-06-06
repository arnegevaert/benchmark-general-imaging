import argparse
import os
import matplotlib as mpl
import numpy as np
from util.get_dataframes import get_dataframes
from krippendorff import krippendorff
from scipy.stats import rankdata
import warnings
from util.plot import plot_krippendorff_alpha
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt


DATASET_COLORS = {
    "MNIST": "#036d08",
    "FashionMNIST": "#9de78c",
    "CIFAR-10": "#08036d",
    "CIFAR-100": "#845eb3",
    "SVHN": "#e7c3ff",
    "ImageNet": "#6d012a",
    "Caltech": "#b65a73",
    "Places": "#ffaac4",
}


def get_dataframes_from_datasets(
    in_dir: str, metric_selection: str, baseline: str
):
    dfs = {}
    for ds_name in DATASET_COLORS.keys():
        if os.path.isdir(os.path.join(in_dir, ds_name)):
            dfs[ds_name] = get_dataframes(
                os.path.join(in_dir, ds_name),
                metric_selection,
                baseline=baseline,
            )
        else:
            warnings.warn(
                f"Results for dataset {ds_name} not found in {in_dir}"
            )
    return dfs


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--in-dir", type=str, default="out")
    parser.add_argument("-o", "--out-dir", type=str, default="plot")
    args = parser.parse_args()

    mpl.use("Agg")
    np.seterr(all="raise")

    for metric_selection in ("all", "default"):
        # dataset_name -> metric_name -> krippendorff alpha
        dfs = get_dataframes_from_datasets(
            args.in_dir, metric_selection, baseline="Random"
        )

        # Compute Krippendorff Alpha for each metric in each dataset
        k_a = {
            ds_name: {
                metric_name: krippendorff.alpha(
                    rankdata(df.to_numpy(), axis=1),
                    level_of_measurement="ordinal",
                )
                for metric_name, (df, _) in dfs.items()
            }
            for ds_name, dfs in dfs.items()
        }

        sns.set()
        colors = [DATASET_COLORS[ds_name] for ds_name in k_a.keys()]
        fig = plot_krippendorff_alpha(pd.DataFrame(k_a), color=colors)

        fig.savefig(
            os.path.join(
                args.out_dir,
                f"krippendorff_alpha_{metric_selection}.svg",
            ),
            bbox_inches="tight",
        )
        plt.close(fig)
