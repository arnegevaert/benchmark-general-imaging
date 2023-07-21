import argparse
import matplotlib as mpl
import numpy as np
from util.get_dataframes import get_dataframes
from matplotlib import pyplot as plt
import os
from attribench.plot import CLESPlot


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("method1", type=str)
    parser.add_argument("method2", type=str)
    parser.add_argument("-i", "--in-dir", type=str)
    parser.add_argument("-o", "--out-dir", type=str)
    args = parser.parse_args()

    mpl.use("Agg")
    np.seterr(all="raise")

    dfs = get_dataframes(args.in_dir, mode="all")
    cles_plot = CLESPlot(dfs)
    fig = cles_plot.render(args.method1, args.method2)
    fig.savefig(
        os.path.join(
            args.out_dir,
            f"pairwise_{args.method1}_{args.method2}.svg",
        ),
        bbox_inches="tight",
    )
    plt.close(fig)
