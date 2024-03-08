import os
from typing import Dict
import pandas as pd
import warnings
import matplotlib as mpl
from matplotlib import pyplot as plt
from attribench import plot
from ..get_dataframes import get_dataframes
import numpy as np
from krippendorff import krippendorff
from scipy.stats import rankdata
import seaborn as sns


def _generate_plot(
    k_a: pd.DataFrame,
    color=None,
    color_thresh=None,
    groups_per_row=15,
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


def generate_krippendorff_alpha_bar_plot(
    in_dir: str,
    out_dir: str,
    dataset_colors: Dict[str, str],
    data_type="image",
):
    mpl.use("Agg")
    np.seterr(all="raise")

    for metric_selection in ("all", "default"):
        # Get dataframes for all datasets
        dfs = {}
        for ds_name in dataset_colors.keys():
            if os.path.isdir(os.path.join(in_dir, ds_name)):
                dfs[ds_name] = get_dataframes(
                    os.path.join(in_dir, ds_name),
                    metric_selection,
                    baseline="Random",
                    data_type=data_type,
                    include_pr=True,
                )
            else:
                warnings.warn(
                    f"Results for dataset {ds_name} not found in {in_dir}"
                )

        # Compute Krippendorff Alpha for each metric in each dataset
        # dataset_name -> metric_name -> krippendorff alpha
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
        colors = [dataset_colors[ds_name] for ds_name in k_a.keys()]
        k_a = pd.DataFrame(k_a)

        # Generate the figure
        fig = _generate_plot(pd.DataFrame(k_a), color=colors)

        fig.savefig(
            os.path.join(
                out_dir, f"krippendorff_alpha_{metric_selection}.svg"
            ),
            bbox_inches="tight",
        )
        plt.close(fig)
        sns.reset_orig()


def generate_krippendorff_alpha_bar_plot_single_dataset(
    in_dir: str,
    out_file: str,
    data_type="image",
    color_thresh=0.5,
    metric_order=None,
    metric_selection="all",
):
    mpl.use("Agg")
    np.seterr(all="raise")

    dfs = get_dataframes(
        in_dir,
        metric_selection,
        baseline="Random",
        data_type=data_type,
        include_pr=True,
    )

    # Compute Krippendorff Alpha for each metric in each dataset
    # dataset_name -> metric_name -> krippendorff alpha
    k_a = {
        metric_name: krippendorff.alpha(
            rankdata(df.to_numpy(), axis=1),
            level_of_measurement="ordinal",
        )
        for metric_name, (df, _) in dfs.items()
    }

    sns.set_theme()
    k_a = pd.DataFrame(k_a, index=["alpha"]).transpose()

    if metric_order is not None:
        metric_order = [m for m in metric_order if m in k_a.index]
        k_a = k_a.reindex(metric_order)

    # Generate the figure
    fig = _generate_plot(k_a, color_thresh=color_thresh, groups_per_row=14)

    fig.savefig(os.path.join(out_file), bbox_inches="tight")
    plt.close(fig)
    sns.reset_orig()