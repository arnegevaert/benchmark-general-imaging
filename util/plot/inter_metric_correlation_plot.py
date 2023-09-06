import matplotlib as mpl
import numpy as np
from ..get_dataframes import get_dataframes
from attribench import plot
from attribench.result import MetricResult, MinimalSubsetResult
from matplotlib import pyplot as plt
import os
from typing import List


def generate_inter_metric_correlation_plot(
    in_dir: str, out_file: str, data_type="image"
):
    """Generates inter-metric correlation plot using default dataframes in
    `in_dir`.
    """
    mpl.use("Agg")
    np.seterr(all="raise")

    dfs = get_dataframes(in_dir, mode="default", data_type=data_type)
    fig = plot.InterMetricCorrelationPlot(dfs).render(
        figsize=(7, 7),
        annot=False,
        fontsize=15,
    )
    fig.savefig(out_file, bbox_inches="tight")
    plt.close(fig)


def generate_inter_metric_correlation_plot_with_maskers(
    in_dir: str, out_file: str, metric: str
):
    """Generates inter-metric correlation plot for a single metric using all
    available maskers."""
    mpl.use("Agg")
    np.seterr(all="raise")

    result_path = os.path.join(in_dir, metric + ".h5")
    if not os.path.exists(result_path):
        result_path = os.path.join(in_dir, metric)
    result = MetricResult.load(result_path)

    maskers = result.levels["masker"]
    if isinstance(result, MinimalSubsetResult):
        dfs = {masker: result.get_df(masker=masker) for masker in maskers}
    else:
        dfs = {
            masker: result.get_df(masker=masker, activation_fn="linear")
            for masker in maskers
        }

    fig = plot.InterMetricCorrelationPlot(dfs).render(
        figsize=(3, 3), annot=True, fontsize=15
    )

    ax = fig.axes[0]
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)

    fig.savefig(out_file, bbox_inches="tight")
    plt.close(fig)


def generate_avg_inter_metric_correlation_plot(
    in_dir: str, out_file: str, datasets: List[str]
):
    mpl.use("Agg")
    np.seterr(all="raise")

    dfs = {
        ds_name: get_dataframes(os.path.join(in_dir, ds_name), mode="default")
        for ds_name in datasets
    }
    fig = plot.AvgInterMetricCorrelationPlot(dfs).render(
        figsize=(7, 7),
        annot=False,
        fontsize=15,
    )
    fig.savefig(out_file, bbox_inches="tight")
    plt.close(fig)
