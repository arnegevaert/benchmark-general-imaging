"""
This script generates a grouped bar plot of the Krippendorff Alpha values
for all metrics and datasets of which the raw results are provided.

The '--in-dir' argument should point to a directory containing the results
for all datasets, i.e. the directory should contain a subdirectory for each
dataset. Each of these subdirectories should contain the results for the
corresponding dataset, as produced by the 'scripts/benchmark/3_run_benchmark.py'
script.

The '--out-dir' argument should point to the directory where the plots should
be saved. The script will write two files to this directory:
'krippendorff_alpha_all.svg' and 'krippendorff_alpha_default.svg'. The former
contains the Krippendorff Alpha values for all metrics, the latter only for
the default metrics.
"""
import argparse
import os
import matplotlib as mpl
import numpy as np
from util.get_dataframes import get_dataframes
from krippendorff import krippendorff
from scipy.stats import rankdata
import warnings
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


def generate_plot(
    k_a: pd.DataFrame,
    color=None,
    color_thresh=None,
    groups_per_row=14,
    legend=True,
):
    if color is None and color_thresh is None:
        raise ValueError("Specify color or color_thresh")

    num_rows = (k_a.shape[0] - 1) // groups_per_row + 1
    fig, axs = plt.subplots(
        nrows=num_rows, figsize=(10, 4 * num_rows), constrained_layout=True
    )

    if num_rows == 1:
        axs = [axs]

    for i in range(num_rows):
        end = min((i + 1) * groups_per_row, k_a.shape[0])
        part = k_a[i * groups_per_row : end]
        cur_axs = axs[i]
        assert isinstance(cur_axs, plt.Axes)

        if color is None:
            color_arr = np.where(part["alpha"] > color_thresh, "g", "r")
            part.plot.bar(
                y="alpha",
                use_index=True,
                ax=cur_axs,
                color=list(color_arr),
                width=0.7,
            )
        else:
            part.plot.bar(ax=cur_axs, color=color, width=0.7)

        cur_axs.set_xticklabels(
            cur_axs.get_xticklabels(),
            rotation=45,
            ha="right",
            rotation_mode="anchor",
        )

        if i > 0 or not legend:
            cur_axs.legend().set_visible(False)
        else:
            cur_axs.legend(
                loc="upper center",
                bbox_to_anchor=(0.5, 1.4),
                fancybox=True,
                shadow=True,
                ncol=4,
            )
        plt.grid(axis="x")
    return fig


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
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "-i",
        "--in-dir",
        type=str,
        help="Path to results directory.",
        required=True,
    )
    parser.add_argument(
        "-o",
        "--out-dir",
        type=str,
        help="Path to output directory.",
        required=True,
    )
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
        fig = generate_plot(pd.DataFrame(k_a), color=colors)

        fig.savefig(
            os.path.join(
                args.out_dir,
                f"krippendorff_alpha_{metric_selection}.svg",
            ),
            bbox_inches="tight",
        )
        plt.close(fig)
