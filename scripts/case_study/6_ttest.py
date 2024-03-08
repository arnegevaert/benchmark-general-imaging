import argparse
from util.get_dataframes import get_dataframes
from attribench import plot
from matplotlib import pyplot as plt
import os


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("in_dir", type=str)
    parser.add_argument("out_dir", type=str)
    args = parser.parse_args()

    ##############################
    # SIGNIFICANCE SUMMARY PLOTS #
    ##############################
    method_order = [
        "DeepSHAP",
        "ExpectedGradients",
        "DeepLIFT",
        "GradCAM",
        "KernelSHAP",
        "LIME",
        "SmoothGrad",
        "IntegratedGradients",
        "InputXGradient",
        "Gradient",
    ]
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
    dfs = get_dataframes(
        args.in_dir, "all", baseline="Random", data_type="image"
    )
    dfs_metric_filtered = {key: dfs[key] for key in metric_order}
    dfs_method_filtered = {
        key: (df[method_order], higher_is_better)
        for key, (df, higher_is_better) in dfs_metric_filtered.items()
    }
    fig = plot.SignificanceSummaryPlot(dfs_method_filtered).render(
        figsize=(8, 8),
        glyph_scale=800,
        fontsize=25,
        method_order=method_order,
        multiple_testing="bonferroni",
        test="t_test",
    )
    fig.savefig(
        os.path.join(args.out_dir, f"ttest.svg"),
        bbox_inches="tight",
    )
    plt.close(fig)
