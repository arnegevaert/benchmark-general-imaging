import argparse
from attrbench.suite import SuiteResult
from attrbench.suite.plot import WilcoxonSummaryPlot
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from os import path
import os
import glob
from dfs import get_default_dfs, get_all_dfs
from tqdm import tqdm


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("in_dir", type=str)
    parser.add_argument("out_dir", type=str)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--mode", type=str, choices=["single", "raw"], default="single")
    args = parser.parse_args()

    mpl.use("Agg")
    np.seterr(all="raise")

    if not path.isdir(args.out_dir):
        os.makedirs(args.out_dir)

    result_files = glob.glob(path.join(args.in_dir, "*.h5"))
    prog = tqdm(result_files)
    for file in prog:
        basename = path.basename(file)
        ds_name, ext = basename.split(".")
        prog.set_postfix_str(ds_name)

        res_obj = SuiteResult.load_hdf(file)
        if args.all:
            dfs = get_all_dfs(res_obj, mode=args.mode)
        else:
            dfs = get_default_dfs(res_obj, mode=args.mode)
        figsize = (10, 25) if args.all else (10, 10)
        glyph_scale = 1000
        fig = WilcoxonSummaryPlot(dfs).render(figsize=figsize, glyph_scale=glyph_scale, fontsize=25)
        fig.savefig(path.join(args.out_dir, f"{ds_name}.png"), bbox_inches="tight")
        plt.close(fig)
