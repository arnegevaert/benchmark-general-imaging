import argparse
import seaborn as sns
import pandas as pd
from tqdm import tqdm
import matplotlib as mpl
import numpy as np
import os
from attrbench.suite import SuiteResult
from util.wilcoxon import wilcoxon
from util.krip import compute_krippendorff_alpha, plot_krippendorff_alpha
from util.inter_metric_correlations import inter_metric_correlations
from matplotlib import pyplot as plt
from util.pairwise_tests import pairwise_tests
from attrbench.suite.plot import InterMetricCorrelationPlot


def _create_out_dir(dirname=""):
    if not os.path.isdir(os.path.join(args.out_dir, dirname)):
        os.makedirs(os.path.join(args.out_dir, dirname))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("in_dir", type=str)
    parser.add_argument("out_dir", type=str)
    args = parser.parse_args()

    mpl.use("Agg")
    np.seterr(all="raise")

    _create_out_dir()
    datasets = ["mnist", "fashionmnist", "cifar10", "cifar100", "svhn", "imagenet", "caltech", "places"]
    result_objects = {
        os.path.basename(filename).split(".")[0]: SuiteResult.load_hdf(filename)
        for filename in [os.path.join(args.in_dir, f"{ds}.h5") for ds in datasets]
    }

    ######################
    # KRIPPENDORFF ALPHA #
    ######################
    sns.set()
    for metric_selection in ("all", "default"):
        k_a = {
            ds_name: compute_krippendorff_alpha(res, metric_selection=metric_selection)
            for ds_name, res in tqdm(result_objects.items())
        }
        k_a = pd.DataFrame(k_a)
        fig = plot_krippendorff_alpha(k_a, ["#036d08", "#9de78c", "#08036d", "#845eb3", "#e7c3ff", "#6d012a", "#b65a73", "#ffaac4"])
        fig.savefig(os.path.join(args.out_dir, f"krip_bar_{metric_selection}.png"), bbox_inches="tight", dpi=250)
    sns.reset_orig()

    ############
    # WILCOXON #
    ############
    for metric_selection in ("default", "all"):
        _create_out_dir(f"wilcoxon_{metric_selection}")
        for ds_name, res_obj in tqdm(result_objects.items()):
            fig = wilcoxon(res_obj, metric_selection)
            fig.savefig(os.path.join(args.out_dir, f"wilcoxon_{metric_selection}", f"{ds_name}.png"), bbox_inches="tight", dpi=250)
            plt.close(fig)

    #############################
    # INTER-METRIC CORRELATIONS #
    #############################
    for metric_selection in ("default", "all"):
        _create_out_dir(f"metric_corr_{metric_selection}")
        for ds_name, res_obj in tqdm(result_objects.items()):
            fig = inter_metric_correlations(res_obj, metric_selection=metric_selection)
            fig.savefig(os.path.join(args.out_dir, f"metric_corr_{metric_selection}", f"{ds_name}.png"),
                        bbox_inches="tight", dpi=250)
            plt.close(fig)

    ##################
    # PAIRWISE TESTS #
    ##################
    for ds_name in tqdm(["mnist", "cifar10", "imagenet"]):
        fig = pairwise_tests(result_objects[ds_name], "DeepSHAP", "DeepLIFT", title="P(DeepSHAP > DeepLIFT)")
        fig.savefig(os.path.join(args.out_dir, f"cles_{ds_name}.png"), bbox_inches="tight", dpi=250)
        plt.close(fig)

    ##################################
    # GENERAL COMPARISON ON IMAGENET #
    ##################################
    res_obj = SuiteResult.load_hdf(os.path.join(args.in_dir, "imagenet.h5"))
    metric_names = ["deletion_morf", "deletion_lerf", "insertion_morf", "insertion_lerf", "irof_morf", "irof_lerf",
                    "minimal_subset_deletion", "minimal_subset_insertion"]
    _create_out_dir("metric_corr_masking")
    prog = tqdm(total=len(metric_names) + 6)
    for metric_name in metric_names:
        dfs = {}
        for masker in ["constant", "blur", "random"]:
            dfs[masker] = res_obj.metric_results[metric_name].get_df(
                masker=masker, activation_fn="linear", mode="raw", normalize=True)

        fig = InterMetricCorrelationPlot(dfs).render(figsize=(2,2))
        fig.savefig(os.path.join(args.out_dir, "metric_corr_masking", f"{metric_name}.png"), bbox_inches="tight", dpi=250)
        plt.close(fig)
        prog.update(1)

    ########################
    # SENS-N VS SEG-SENS-N #
    ########################
    for ds_name in ["mnist", "cifar10", "imagenet"]:
        res_obj = SuiteResult.load_hdf(os.path.join(args.in_dir, f"{ds_name}.h5"))
        for metric_name in ["sensitivity_n", "seg_sensitivity_n"]:
            dfs = {}
            for masker in ["constant", "blur", "random"]:
                dfs[masker] = res_obj.metric_results[metric_name].get_df(
                    masker=masker, activation_fn="linear", mode="raw")

            fig = InterMetricCorrelationPlot(dfs).render(figsize=(2, 2))
            fig.savefig(os.path.join(args.out_dir, "metric_corr_masking", f"{ds_name}_{metric_name}.png"), bbox_inches="tight", dpi=250)
            plt.close(fig)
            prog.update(1)
