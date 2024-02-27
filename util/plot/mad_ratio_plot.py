import matplotlib as mpl
import numpy as np
from ..get_dataframes import get_dataframes
from attribench import plot
from matplotlib import pyplot as plt
from attribench.plot import MADRatioPlot


def generate_mad_ratio_plot(in_dir: str, out_file: str):
    mpl.use("Agg")
    np.seterr(all="raise")

    dfs = get_dataframes(in_dir, mode="default")
    fig = MADRatioPlot(dfs).render()
    fig.savefig(out_file, bbox_inches="tight")
    plt.close(fig)
