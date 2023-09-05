import matplotlib as mpl
import numpy as np
from ..get_dataframes import get_dataframes
from attribench import plot
from matplotlib import pyplot as plt
import os


def generate_convergence_plot(in_dir: str, out_dir: str):
    mpl.use("Agg")
    np.seterr(all="raise")

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    dfs = get_dataframes(in_dir, mode="default")
    for metric_name, (df, _) in dfs.items():
        fig = plot.ConvergencePlot(df).render()
        fig.savefig(
            os.path.join(out_dir, metric_name + ".svg"),
            bbox_inches="tight",
        )
        plt.close(fig)
