import argparse
import matplotlib as mpl
from attrbench.suite import SuiteResult
from os import path
import os
from attrbench.suite.plot import InterMetricCorrelationPlot
import matplotlib.pyplot as plt
from tqdm import tqdm


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("in_dir", type=str)
    parser.add_argument("out_dir", type=str)
    args = parser.parse_args()

    mpl.use("Agg")

    if not path.isdir(args.out_dir):
        os.makedirs(path.join(args.out_dir, "general"))
        os.makedirs(path.join(args.out_dir, "sens_n"))

    ##################################
    # GENERAL COMPARISON ON IMAGENET #
    ##################################

    res_obj = SuiteResult.load_hdf(path.join(args.in_dir, "imagenet.h5"))
    metric_names = ["deletion_morf", "deletion_lerf", "insertion_morf", "insertion_lerf", "irof_morf", "irof_lerf",
                    "minimal_subset_deletion", "minimal_subset_insertion"]
    prog = tqdm(total=len(metric_names) + 6)
    for metric_name in metric_names:
        dfs = {}
        for masker in ["constant", "blur", "random"]:
            dfs[masker] = res_obj.metric_results[metric_name].get_df(
                masker=masker, activation_fn="linear", mode="raw", normalize=True)

        fig = InterMetricCorrelationPlot(dfs).render(figsize=(2,2))
        fig.savefig(path.join(args.out_dir, "general", f"{metric_name}.png"), bbox_inches="tight", dpi=250)
        plt.close(fig)
        prog.update(1)

    ########################
    # SENS-N VS SEG-SENS-N #
    ########################

    for ds_name in ["mnist", "cifar10", "imagenet"]:
        res_obj = SuiteResult.load_hdf(path.join(args.in_dir, f"{ds_name}.h5"))
        for metric_name in ["sensitivity_n", "seg_sensitivity_n"]:
            dfs = {}
            for masker in ["constant", "blur", "random"]:
                dfs[masker] = res_obj.metric_results[metric_name].get_df(
                    masker=masker, activation_fn="linear", mode="raw")

            fig = InterMetricCorrelationPlot(dfs).render(figsize=(2, 2))
            fig.savefig(path.join(args.out_dir, "sens_n", f"{ds_name}_{metric_name}.png"), bbox_inches="tight", dpi=250)
            plt.close(fig)
            prog.update(1)
