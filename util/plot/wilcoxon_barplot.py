import matplotlib as mpl
import numpy as np
from ..get_dataframes import get_dataframes
from attribench import plot
from matplotlib import pyplot as plt
import os


def generate_wilcoxon_barplot(in_dir: str, out_dir: str):
    mpl.use("Agg")
    np.seterr(all="raise")

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    for metric_selection in ("default", "all"):
        dfs = get_dataframes(in_dir, mode=metric_selection, baseline="Random")
        wilcoxon_barplot = plot.WilcoxonBarPlot(dfs)
        fig = wilcoxon_barplot.render()
        fig.savefig(
            os.path.join(out_dir, f"wilcoxon_bar_{metric_selection}.svg"),
            bbox_inches="tight",
        )
        plt.close(fig)
