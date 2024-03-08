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
        "Del - MoRF - blurring",
        "Del - MoRF - constant",
        "Del - MoRF - random",
        "Del - LeRF - blurring",
        "Del - LeRF - constant",
        "Del - LeRF - random",
        "Ins - MoRF - blurring",
        "Ins - MoRF - constant",
        "Ins - MoRF - random",
        "IROF - MoRF - blurring",
        "IROF - MoRF - constant",
        "IROF - MoRF - random",
        "IROF - LeRF - blurring",
        "IROF - LeRF - constant",
        "IROF - LeRF - random",
        "SegSensN - blurring",
        "SegSensN - constant",
        "SegSensN - random",
        "INFD - SQ",
        "INFD - BL",
        "MSIns - blurring",
        "MSIns - constant",
        "MSIns - random",
        "MSDel - blurring",
        "MSDel - constant",
        "MSDel - random",
    ]
    dfs = get_dataframes(args.in_dir, mode="all", data_type="image")
    dfs_filtered = {key: dfs[key] for key in metric_order}

    fig = plot.InterMetricCorrelationPlot(dfs_filtered).render(
        figsize=(10, 10),
        annot=False,
        fontsize=15,
    )
    fig.savefig(
        os.path.join(args.out_dir, "metric_corr.svg"), bbox_inches="tight"
    )
    plt.close(fig)
