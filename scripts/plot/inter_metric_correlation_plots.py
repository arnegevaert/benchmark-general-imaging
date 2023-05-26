import os
from attrbench.plot import InterMetricCorrelationPlot
import argparse
import matplotlib as mpl
from matplotlib import pyplot as plt
import numpy as np
from util.get_dataframes import get_dataframes


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--in-dir", type=str, default="out/MNIST")
    parser.add_argument("-o", "--out-dir", type=str, default="plot/MNIST")
    args = parser.parse_args()

    mpl.use("Agg")
    np.seterr(all="raise")

    if not os.path.isdir(os.path.join(args.out_dir, "metric_corr")):
        os.makedirs(os.path.join(args.out_dir, "metric_corr"))

    for metric_selection in ("default", "all"):
        dfs = get_dataframes(args.in_dir, metric_selection)
        fig = InterMetricCorrelationPlot(dfs).render(
            figsize=(7, 7) if metric_selection == "default" else (17, 17),
            annot=metric_selection == "all",
            fontsize=15 if metric_selection == "all" else 17,
        )
        fig.savefig(
            os.path.join(
                args.out_dir,
                "metric_corr",
                f"metric_corr_{metric_selection}.svg",
            ),
            bbox_inches="tight",
        )
        plt.close(fig)
