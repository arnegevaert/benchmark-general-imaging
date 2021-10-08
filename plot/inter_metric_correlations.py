import argparse
from attrbench.suite import SuiteResult
from attrbench.suite.plot import InterMetricCorrelationPlot
from .dfs import get_default_dfs, get_all_dfs
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from os import path
import os
import glob
from tqdm import tqdm


def inter_metric_correlations(result_files, out_dir, metric_selection="default"):
    prog = tqdm(result_files)
    for file in prog:
        basename = path.basename(file)
        ds_name, ext = basename.split(".")
        prog.set_postfix_str(ds_name)
        mode = "raw"
        res_obj = SuiteResult.load_hdf(file)
        if all:
            dfs = get_all_dfs(res_obj, mode=mode)
        else:
            dfs = get_default_dfs(res_obj, mode=mode)

        figsize = (17, 17) if metric_selection == "all" else (7, 7)
        fontsize = 15 if metric_selection == "all" else 17
        fig = InterMetricCorrelationPlot(dfs).render(figsize=figsize, annot=metric_selection == "all")
        ax = fig.axes[0]
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor", fontsize=fontsize)
        ax.set_yticklabels(ax.get_yticklabels(), fontsize=fontsize)
        fig.savefig(path.join(out_dir, f"{ds_name}.png"), bbox_inches="tight", dpi=250)
        plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("in_dir", type=str)
    parser.add_argument("out_dir", type=str)
    parser.add_argument("--all", action="store_true")
    args = parser.parse_args()

    mpl.use("Agg")
    np.seterr(all="raise")

    if not path.isdir(args.out_dir):
        os.makedirs(args.out_dir)

    result_files = glob.glob(path.join(args.in_dir, "*.h5"))
    inter_metric_correlations(result_files, args.out_dir, args.all)
