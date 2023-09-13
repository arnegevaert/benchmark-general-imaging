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
            "cles",
            "krippendorff",
            "param_randomization",
        ],
    )
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
    if "wilcoxon" in args.plots:
        wilcoxon_out_dir = os.path.join(args.out_dir, "wilcoxon")
        if not os.path.isdir(wilcoxon_out_dir):
            os.makedirs(wilcoxon_out_dir)

        prog = tqdm(os.listdir(args.in_dir))
        prog.set_description("Generating Wilcoxon summary plots")
        for dataset in prog:
            in_dir = os.path.join(args.in_dir, dataset)
            plot.generate_wilcoxon_summary_plots(
                in_dir, wilcoxon_out_dir, method_order, dataset
            )

    #############################
    # INTER-METRIC CORRELATIONS #
    #############################
    if "corr" in args.plots:
        # Average for low-/medium-/high-dimensional datasets
        corr_out_dir = os.path.join(args.out_dir, "metric_corr")
        if not os.path.isdir(corr_out_dir):
            os.makedirs(corr_out_dir)
        dataset_groups = {
            "low_dim": ["MNIST", "FashionMNIST"],
            "medium_dim": ["CIFAR10", "CIFAR100", "SVHN"],
            "high_dim": ["ImageNet", "Places365", "Caltech256"],
        }
        prog = tqdm(dataset_groups.items())
        prog.set_description("Generating averaged inter-metric correlation plots")
        for key, datasets in prog:
            plot.generate_avg_inter_metric_correlation_plot(
                args.in_dir, os.path.join(corr_out_dir, key + ".svg"), datasets
            )

        # Per-dataset
        prog = tqdm(os.listdir(args.in_dir))
        prog.set_description("Generating inter-metric correlation plots")
        for dataset in prog:
            in_dir = os.path.join(args.in_dir, dataset)
            plot.generate_inter_metric_correlation_plot(
                in_dir, corr_out_dir, dataset, data_type="image"
            )

    ########################
    # MASKING CORRELATIONS #
    ########################
    if "corr" in args.plots:
        corr_out_dir = os.path.join(args.out_dir, "metric_corr")
        prog = tqdm(
            [
                "deletion_morf",
                "deletion_lerf",
                "insertion_morf",
                "insertion_lerf",
                "irof_morf",
                "irof_lerf",
                "ms_deletion",
                "ms_insertion",
            ]
        )
        prog.set_description("Generating masking correlation plots")
        for metric_name in prog:
            plot.generate_inter_metric_correlation_plot_with_maskers(
                os.path.join(args.in_dir, "ImageNet"),
                os.path.join(corr_out_dir, metric_name + ".svg"),
                metric_name,
            )

    ##############
    # CLES PLOTS #
    ##############
    if "cles" in args.plots:
        prog = tqdm(["MNIST", "CIFAR10", "ImageNet"])
        prog.set_description("Generating CLES plots")
        for dataset_name in prog:
            plot.generate_cles_plot(
                "DeepSHAP",
                "DeepLIFT",
                os.path.join(args.in_dir, dataset_name),
                os.path.join(args.out_dir, f"cles_{dataset_name}.svg"),
            )

    ############################
    # KRIPPENDORFF ALPHA PLOTS #
    ############################
    if "krippendorff" in args.plots:
        dataset_colors = {
            "MNIST": "#036d08",
            "FashionMNIST": "#9de78c",
            "CIFAR10": "#08036d",
            "CIFAR100": "#845eb3",
            "SVHN": "#e7c3ff",
            "ImageNet": "#6d012a",
            "Caltech256": "#b65a73",
            "Places365": "#ffaac4",
        }
        print("Generating Krippendorff Alpha plots")
        plot.generate_krippendorff_alpha_bar_plot(
            args.in_dir, args.out_dir, dataset_colors
        )

    #################################
    # PARAMETER RANDOMIZATION PLOTS #
    #################################
    if "param_randomization" in args.plots:
        results = {}
        for ds_name in os.listdir(args.in_dir):
            metric_result = MetricResult.load(
                os.path.join(args.in_dir, ds_name, "parameter_randomization.h5")
            )
            df, _ = metric_result.get_df()
            results[ds_name] = df.mean()
        result_df = pd.DataFrame.from_dict(results, orient="index")
        result_df.rename(
            columns={"DeepShap": "DeepSHAP", "DeepLift": "DeepLIFT"}, inplace=True
        )

        result_df = result_df.reindex(
            [
                "MNIST",
                "FashionMNIST",
                "CIFAR10",
                "CIFAR100",
                "SVHN",
                "ImageNet",
                "Places365",
                "Caltech256",
            ]
        )
        result_df = result_df[method_order].abs()

        fig, ax = plt.subplots(figsize=(12, 7))
        sns.heatmap(
            result_df,
            annot=True,
            ax=ax,
            cmap=sns.color_palette("RdYlGn_r", 1000),
            fmt=".2f",
            vmin=0,
            vmax=1,
        )
        ax.set_xticklabels(
            ax.get_xticklabels(), rotation=45, horizontalalignment="right"
        )
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
        fig.savefig(
            os.path.join(args.out_dir, "parameter_randomization.svg"),
            bbox_inches="tight",
        )
