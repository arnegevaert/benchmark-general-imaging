import argparse
import matplotlib as mpl
import numpy as np
from util.get_dataframes import get_dataframes
from attribench.plot import ConvergencePlot
import os
from matplotlib import pyplot as plt


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--in-dir", type=str)
    parser.add_argument("-o", "--out-dir", type=str)
    args = parser.parse_args()

    mpl.use("Agg")
    np.seterr(all="raise")

    out_dir = os.path.join(args.out_dir, "convergence")

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    dfs = get_dataframes(args.in_dir, mode="default")
    for metric_name, (df, higher_is_better) in dfs.items():
        fig = ConvergencePlot(df).render()
        fig.savefig(
            os.path.join(out_dir, metric_name + ".svg"),
            bbox_inches="tight",
        )
        plt.close(fig)
