import os
from attribench.plot import InterMetricCorrelationPlot
import argparse
import matplotlib as mpl
from matplotlib import pyplot as plt
import numpy as np
from util.get_dataframes import get_dataframes


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--in-dir", type=str)
    parser.add_argument("-o", "--out-dir", type=str)
    args = parser.parse_args()

    mpl.use("Agg")
    np.seterr(all="raise")

    dfs = get_dataframes(args.in_dir, mode="default")
    fig = InterMetricCorrelationPlot(dfs).render(
        figsize=(7, 7),
        annot=False,
        fontsize=15,
    )
    fig.savefig(
        os.path.join(
            args.out_dir,
            "metric_corr.svg",
        ),
        bbox_inches="tight",
    )
    plt.close(fig)
