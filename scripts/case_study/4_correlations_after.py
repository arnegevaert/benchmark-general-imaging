import argparse
from util import plot
import os
from util.get_dataframes import get_dataframes
from attribench import plot
from matplotlib import pyplot as plt


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("in_dir", type=str)
    parser.add_argument("out_dir", type=str)
    args = parser.parse_args()

    metric_order = [
        "Cov",
        "MaxSens",
        "Ins - MoRF - random",
        "IROF - MoRF - constant",
        "IROF - LeRF - random",
        "SegSensN - constant",
        "INFD - SQ",
        "INFD - BL",
        "MSIns - random",
        "MSDel - constant",
    ]
    dfs = get_dataframes(args.in_dir, mode="all", data_type="image")
    dfs_filtered = {key: dfs[key] for key in metric_order}

    fig = plot.InterMetricCorrelationPlot(dfs_filtered).render(
        figsize=(7, 7),
        annot=True,
        fontsize=15,
    )
    fig.savefig(
        os.path.join(args.out_dir, "metric_corr_selected.svg"), bbox_inches="tight"
    )
    plt.close(fig)
