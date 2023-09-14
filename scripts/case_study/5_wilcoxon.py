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

    ##########################
    # WILCOXON SUMMARY PLOTS #
    ##########################
    method_order = [
        "DeepSHAP",
        "ExpectedGradients",
        "DeepLIFT",
        "GradCAM",
        "GradCAM++",
        "ScoreCAM",
        "XRAI",
        "KernelSHAP",
        "LIME",
        "SmoothGrad",
        "VarGrad",
        "IntegratedGradients",
        "InputXGradient",
        "Gradient",
        "GuidedBackprop",
        "GuidedGradCAM",
        "Deconvolution",
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
    dfs_filtered = {key: dfs[key] for key in metric_order}
    fig = plot.WilcoxonSummaryPlot(dfs_filtered).render(
        figsize=(12, 10),
        glyph_scale=800,
        fontsize=25,
        method_order=method_order,
        multiple_testing="bonferroni",
    )
    fig.savefig(
        os.path.join(args.out_dir, f"wilcoxon.svg"),
        bbox_inches="tight",
    )
    plt.close(fig)