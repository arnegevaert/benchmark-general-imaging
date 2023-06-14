import argparse
import matplotlib as mpl
import numpy as np
from util.get_dataframes import get_dataframes
from attribench.plot import MADRatioPlot
import os
from matplotlib import pyplot as plt


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i", "--in-dir", type=str, default="out/results/MNIST"
    )
    parser.add_argument("-o", "--out-dir", type=str, default="out/plots/MNIST")
    args = parser.parse_args()

    mpl.use("Agg")
    np.seterr(all="raise")

    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)

    dfs = get_dataframes(args.in_dir, mode="default")
    fig = MADRatioPlot(dfs).render()
    fig.savefig(
        os.path.join(
            args.out_dir,
            "mad_ratio.svg",
        ),
        bbox_inches="tight",
    )
    plt.close(fig)
