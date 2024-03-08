import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from attribench.result import MetricResult
from typing import List, Optional
import matplotlib.colors as mcolors


def generate_parameter_randomization_plot(
    in_dir: str,
    out_dir: str,
    method_order: List[str],
    dataset_order: Optional[List[str]] = None,
):
    results = {}
    for ds_name in os.listdir(in_dir):
        metric_result = MetricResult.load(
            os.path.join(in_dir, ds_name, "parameter_randomization.h5")
        )
        df, _ = metric_result.get_df()
        results[ds_name] = df.mean()
    result_df = pd.DataFrame.from_dict(results, orient="index")
    result_df.rename(
        columns={"DeepShap": "DeepSHAP", "DeepLift": "DeepLIFT"},
        inplace=True,
    )
    if dataset_order is not None:
        result_df = result_df.reindex(dataset_order)
    result_df = result_df[method_order].abs()

    orig_palette = sns.color_palette("RdYlGn_r", 100)
    cmap = mcolors.LinearSegmentedColormap.from_list(
        "cmap",
        [
            (0, orig_palette[0]),
            (0.1, orig_palette[49]),
            (0.2, orig_palette[99]),
            (1, orig_palette[99]),
        ],
    )

    fig, ax = plt.subplots(figsize=(12, 7))
    sns.heatmap(
        result_df,
        annot=True,
        ax=ax,
        cmap=cmap,
        fmt=".2f",
        vmin=0,
        vmax=1,
    )
    ax.set_xticklabels(
        ax.get_xticklabels(), rotation=45, horizontalalignment="right"
    )
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
    fig.savefig(
        os.path.join(out_dir, "parameter_randomization.svg"),
        bbox_inches="tight",
    )
