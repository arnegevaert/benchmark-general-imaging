import argparse
from attrbench.suite import SuiteResult
from attrbench.suite.plot import InterMetricCorrelationPlot, WilcoxonSummaryPlot
import matplotlib as mpl
import numpy as np
import seaborn as sns
from util.krip import compute_krippendorff_alpha, plot_krippendorff_alpha
import os
from pingouin import wilcoxon
from matplotlib import pyplot as plt
import pandas as pd


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("small_file", type=str)
    parser.add_argument("big_file", type=str)
    parser.add_argument("out_dir", type=str)
    args = parser.parse_args()

    mpl.use("Agg")
    np.seterr(all="raise")
    sns.set()

    if not os.path.isdir(args.out_dir):
        os.makedirs(args.out_dir)

    #######################################################
    # STEP 1: SELECT METRICS WITH HIGH KRIPPENDORFF ALPHA #
    #######################################################

    threshold = 0.3
    result_object = SuiteResult.load_hdf(args.small_file)
    k_a = compute_krippendorff_alpha(result_object, metric_selection="all")
    k_a = k_a.to_frame("alpha")
    fig = plot_krippendorff_alpha(k_a, color_thresh=threshold, legend=False)
    fig.savefig(os.path.join(args.out_dir, "krip_bar.png"), bbox_inches="tight", dpi=250)

    ##########################################
    # STEP 2: PLOT INTER-METRIC CORRELATIONS #
    ##########################################

    mr = result_object.metric_results
    high_ka_metrics = {
        "Cov": mr["impact_coverage"].get_df(mode="raw"),
        "MaxSens": mr["max_sensitivity"].get_df(mode="raw"),
        "Del - MoRF - blur": mr["deletion_morf"].get_df(masker="blur", mode="raw", normalize=True),
        "Del - MoRF - constant": mr["deletion_morf"].get_df(masker="constant", mode="raw", normalize=True),
        "Del - MoRF - random": mr["deletion_morf"].get_df(masker="random", mode="raw", normalize=True),
        "Del - LeRF - blur": mr["deletion_lerf"].get_df(masker="blur", mode="raw", normalize=True),
        "Del - LeRF - constant": mr["deletion_lerf"].get_df(masker="constant", mode="raw", normalize=True),
        "Del - LeRF - random": mr["deletion_lerf"].get_df(masker="random", mode="raw", normalize=True),
        "Ins - MoRF - blur": mr["insertion_morf"].get_df(masker="blur", mode="raw", normalize=True),
        "Ins - MoRF - constant": mr["insertion_morf"].get_df(masker="constant", mode="raw", normalize=True),
        "IROF - MoRF - blur": mr["irof_morf"].get_df(masker="blur", mode="raw", normalize=True),
        "IROF - MoRF - constant": mr["irof_morf"].get_df(masker="constant", mode="raw", normalize=True),
        "IROF - MoRF - random": mr["irof_morf"].get_df(masker="random", mode="raw", normalize=True),
        "IROF - LeRF - constant": mr["irof_lerf"].get_df(masker="constant", mode="raw", normalize=True),
        "IROF - LeRF - blur": mr["irof_lerf"].get_df(masker="blur", mode="raw", normalize=True),
        "IROF - LeRF - random": mr["irof_lerf"].get_df(masker="random", mode="raw", normalize=True),
        "SegSensN - blur": mr["seg_sensitivity_n"].get_df(masker="blur", mode="raw"),
        "SegSensN - constant": mr["seg_sensitivity_n"].get_df(masker="constant", mode="raw"),
        "SegSensN - random": mr["seg_sensitivity_n"].get_df(masker="random", mode="raw"),
        "Infidelity - noisy_bl": mr["infidelity"].get_df(perturbation_generator="noisy_bl", mode="raw"),
        "MSIns - blur": mr["minimal_subset_insertion"].get_df(masker="blur", mode="raw"),
        "MSIns - constant": mr["minimal_subset_insertion"].get_df(masker="constant", mode="raw"),
        "MSIns - random": mr["minimal_subset_insertion"].get_df(masker="random", mode="raw"),
        "MSDel - blur": mr["minimal_subset_deletion"].get_df(masker="blur", mode="raw"),
        "MSDel - constant": mr["minimal_subset_deletion"].get_df(masker="constant", mode="raw"),
        "MSDel - random": mr["minimal_subset_deletion"].get_df(masker="random", mode="raw")
    }
    fontsize = 15
    fig = InterMetricCorrelationPlot(high_ka_metrics).render(figsize=(8, 8), annot=False)
    ax = fig.axes[0]
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor", fontsize=fontsize)
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=fontsize)
    fig.savefig(os.path.join(args.out_dir, "metric_corr.png"), bbox_inches="tight", dpi=250)

    selected_metrics = {
        "Cov": mr["impact_coverage"].get_df(mode="raw"),
        "MaxSens": mr["max_sensitivity"].get_df(mode="raw"),
        "Ins - MoRF - constant": mr["insertion_morf"].get_df(masker="constant", mode="raw", normalize=True),
        "IROF - MoRF - constant": mr["irof_morf"].get_df(masker="constant", mode="raw", normalize=True),
        "IROF - LeRF - random": mr["irof_lerf"].get_df(masker="random", mode="raw", normalize=True),
        "SegSensN - constant": mr["seg_sensitivity_n"].get_df(masker="constant", mode="raw"),
        "Infidelity - noisy_bl": mr["infidelity"].get_df(perturbation_generator="noisy_bl", mode="raw"),
        "MSIns - random": mr["minimal_subset_insertion"].get_df(masker="random", mode="raw"),
        "MSDel - constant": mr["minimal_subset_deletion"].get_df(masker="constant", mode="raw"),
    }
    fig = InterMetricCorrelationPlot(selected_metrics).render(figsize=(7, 7), annot=True)
    ax = fig.axes[0]
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor", fontsize=fontsize)
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=fontsize)
    fig.savefig(os.path.join(args.out_dir, "metric_corr_selected.png"), bbox_inches="tight", dpi=250)

    #################################################################
    # STEP 3: WILCOXON SIGNED RANK TESTS ON LARGE AMOUNT OF SAMPLES #
    #################################################################
    large_result_obj = SuiteResult.load_hdf(args.big_file)
    dfs = {
        "Cov": mr["impact_coverage"].get_df(mode="single"),
        "MaxSens": mr["max_sensitivity"].get_df(mode="single"),
        "Ins - MoRF - constant": mr["insertion_morf"].get_df(masker="constant", mode="single", normalize=True),
        "IROF - MoRF - constant": mr["irof_morf"].get_df(masker="constant", mode="single", normalize=True),
        "IROF - LeRF - random": mr["irof_lerf"].get_df(masker="random", mode="single", normalize=True),
        "SegSensN - constant": mr["seg_sensitivity_n"].get_df(masker="constant", mode="single"),
        "Infidelity - noisy_bl": mr["infidelity"].get_df(perturbation_generator="noisy_bl", mode="single"),
        "MSIns - random": mr["minimal_subset_insertion"].get_df(masker="random", mode="single"),
        "MSDel - constant": mr["minimal_subset_deletion"].get_df(masker="constant", mode="single"),
    }
    order = ["DeepShap", "ExpectedGradients", "DeepLift", "GradCAM", "KernelShap", "LIME", "SmoothGrad", "VarGrad",
             "IntegratedGradients", "InputXGradient", "Gradient", "GuidedBackprop", "GuidedGradCAM", "Deconvolution"]
    fig = WilcoxonSummaryPlot(dfs).render(figsize=(8, 8), glyph_scale=800, fontsize=fontsize, method_order=order)
    fig.savefig(os.path.join(args.out_dir, "wilcoxon.png"), bbox_inches="tight", dpi=250)

    ####################################################
    # STEP 4: PAIRWISE COMPARISON BETWEEN BEST METHODS #
    ####################################################
    for pair in [("DeepShap", "GradCAM")]:
        m1, m2 = pair
        result_cles = {}
        for key in dfs:
            df, inverted = dfs[key]
            df = -df if inverted else df

            res = wilcoxon(x=df[m1], y=df[m2], alternative="two-sided")
            pvalue = res["p-val"]["Wilcoxon"]
            cles = res["CLES"]["Wilcoxon"]
            result_cles[key] = cles if pvalue < 0.01 else 0.5
        sns.set_color_codes("muted")
        df = pd.DataFrame(result_cles, index=["CLES"]).transpose().reset_index().rename(columns={"index": "Metric"})
        df["CLES"] -= 0.5
        fig, ax = plt.subplots(figsize=(5, 7))
        sns.barplot(data=df, x="CLES", y="Metric", color="b", left=0.5, ax=ax)
        ax.set(xlim=(0, 1))
        ax.set_title(f"P({m1} > {m2})")
        fig.savefig(os.path.join(args.out_dir, f"cles_{m1}_{m2}.png"), bbox_inches="tight", dpi=250)
