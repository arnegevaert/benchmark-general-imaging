import matplotlib as mpl
import numpy as np
from ..get_dataframes import get_dataframes
from attribench import plot
from matplotlib import pyplot as plt
import os


def generate_cluster_plot(in_dir: str, out_file: str):
    mpl.use("Agg")
    np.seterr(all="raise")

    dfs = get_dataframes(in_dir, mode="default")
    fig = plot.ClusterPlot(dfs).render()
    fig.savefig(os.path.join(out_file), bbox_inches="tight")
    plt.close(fig)