import argparse
import seaborn as sns
import os
from util import plot
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
from attribench.result import MetricResult


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("in_dir", type=str)
    parser.add_argument("out_dir", type=str)
    parser.add_argument(
        "-p",
        "--plots",
        nargs="*",
        default=[
            "wilcoxon",
            "corr",
            "krippendorff",
            "param_randomization",
        ],
    )
    args = parser.parse_args()

    prog = tqdm(args.plots)

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
    if "wilcoxon" in args.plots:
        prog.set_description("Generating Wilcoxon summary plots")
        plot.generate_significance_summary_plots(
            args.in_dir, args.out_dir, method_order, "wilcoxon"
        )
        prog.update()

    #############################
    # INTER-METRIC CORRELATIONS #
    #############################
    if "corr" in args.plots:
        prog.set_description("Generating inter-metric correlation plots")
        plot.generate_inter_metric_correlation_plot(
            args.in_dir, args.out_dir, "metric_corr"
        )
        prog.update()

    ############################
    # KRIPPENDORFF ALPHA PLOTS #
    ############################
    if "krippendorff" in args.plots:
        prog.set_description("Generating Krippendorff Alpha plots")
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
            "Ins - LeRF - blurring",
            "Ins - LeRF - constant",
            "Ins - LeRF - random",
            "IROF - MoRF - blurring",
            "IROF - MoRF - constant",
            "IROF - MoRF - random",
            "IROF - LeRF - blurring",
            "IROF - LeRF - constant",
            "IROF - LeRF - random",
            "SensN - blurring",
            "SensN - constant",
            "SensN - random",
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
        plot.generate_krippendorff_alpha_bar_plot_single_dataset(
            args.in_dir,
            args.out_dir,
            color_thresh=0.5,
            metric_order=metric_order,
        )
        prog.update()

    #################################
    # PARAMETER RANDOMIZATION PLOTS #
    #################################
    if "param_randomization" in args.plots:
        prog.set_description("Generating parameter randomization plots")
        metric_result = MetricResult.load(
            os.path.join(args.in_dir, "parameter_randomization.h5")
        )
        df, _ = metric_result.get_df()

        df.rename(
            columns={"DeepShap": "DeepSHAP", "DeepLift": "DeepLIFT"},
            inplace=True,
        )
        result = df.mean()
        result = result[method_order].abs()

        fig, ax = plt.subplots(figsize=(12, 7))
        result.plot.bar(ax=ax)

        # ax.set_xticklabels(
        #    ax.get_xticklabels(), rotation=45, horizontalalignment="right"
        # )
        # ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
        fig.savefig(
            os.path.join(args.out_dir, "parameter_randomization.svg"),
            bbox_inches="tight",
        )
        prog.update()
    prog.close()
