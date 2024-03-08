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
    args = parser.parse_args()
    
    method_order = [
        "DeepSHAP",
        "ExpectedGradients",
        "DeepLIFT",
        "KernelSHAP",
        "LIME",
        "SmoothGrad",
        "VarGrad",
        "IntegratedGradients",
        "InputXGradient",
        "Gradient",
    ]

    ##############################
    # SIGNIFICANCE SUMMARY PLOTS #
    ##############################
    for test in ["wilcoxon", "sign_test", "t_test"]:
        out_dir = os.path.join(args.out_dir, test)
        if not os.path.isdir(out_dir):
            os.makedirs(out_dir)

        prog = tqdm(os.listdir(args.in_dir))
        prog.set_description(f"Generating {test} summary plots")
        for dataset in prog:
            in_dir = os.path.join(args.in_dir, dataset)
            plot.generate_significance_summary_plots(
                in_dir,
                out_dir,
                method_order,
                dataset,
                data_type="tabular",
                test=test,
            )

    #############################
    # INTER-METRIC CORRELATIONS #
    #############################
    corr_out_dir = os.path.join(args.out_dir, "metric_corr")
    if not os.path.isdir(corr_out_dir):
        os.makedirs(corr_out_dir)

    prog = tqdm(os.listdir(args.in_dir))
    prog.set_description("Generating inter-metric correlation plots")
    for dataset in prog:
        in_dir = os.path.join(args.in_dir, dataset)
        plot.generate_inter_metric_correlation_plot(
            in_dir,
            corr_out_dir,
            dataset,
            data_type="tabular",
        )

    ############################
    # KRIPPENDORFF ALPHA PLOTS #
    ############################
    dataset_colors = {
        "adult": "#036d08",
        "dna": "#08036d",
        "satimage": "#6d012a",
        "spambase": "#d6d6b1",
    }
    print("Generating Krippendorff alpha plots")
    plot.generate_krippendorff_alpha_bar_plot(
        args.in_dir, args.out_dir, dataset_colors, "tabular"
    )

    #################################
    # PARAMETER RANDOMIZATION PLOTS #
    #################################
    print("Generating parameter randomization plot")
    plot.generate_parameter_randomization_plot(
        args.in_dir, args.out_dir, method_order
    )