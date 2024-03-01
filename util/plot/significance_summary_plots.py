import matplotlib as mpl
import numpy as np
from ..get_dataframes import get_dataframes
from attribench import plot
from matplotlib import pyplot as plt
import os
from typing import List


def generate_significance_summary_plots(
    in_dir: str,
    out_dir: str,
    method_order: List[str],
    dataset_name: str,
    glyph_scale=800,
    data_type="image",
    test="wilcoxon",
):
    mpl.use("Agg")
    np.seterr(all="raise")

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    for metric_selection in ("default", "all"):
        dfs = get_dataframes(
            in_dir, metric_selection, baseline="Random", data_type=data_type
        )
        fig = plot.SignificanceSummaryPlot(dfs).render(
            figsize=(10, 10) if metric_selection == "default" else (10, 25),
            glyph_scale=glyph_scale,
            fontsize=25,
            method_order=method_order,
            multiple_testing="bonferroni",
            test=test,
        )
        fig.savefig(
            os.path.join(out_dir, f"{dataset_name}_{metric_selection}.svg"),
            bbox_inches="tight",
        )
        plt.close(fig)
