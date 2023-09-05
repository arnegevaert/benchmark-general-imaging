import matplotlib as mpl
import numpy as np
from ..get_dataframes import get_dataframes
from attribench import plot
from matplotlib import pyplot as plt


def generate_cles_plot(method1: str, method2: str, in_dir: str, out_file: str):
    mpl.use("Agg")
    np.seterr(all="raise")

    dfs = get_dataframes(in_dir, mode="all")
    cles_plot = plot.CLESPlot(dfs)
    fig = cles_plot.render(method1, method2)
    fig.savefig(out_file, bbox_inches="tight")
    plt.close(fig)