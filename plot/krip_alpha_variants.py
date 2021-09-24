import argparse
import os
import pandas as pd
from attrbench.lib import krippendorff_alpha
from attrbench.suite import SuiteResult
from experiments.general_imaging.plot.dfs import get_default_dfs
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from os import path
import glob
import seaborn as sns
from tqdm import tqdm


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("in_dir", type=str)
    parser.add_argument("out_dir", type=str)
    args = parser.parse_args()

    mpl.use("Agg")
    np.seterr(all="raise")
    sns.set()

    if not path.isdir(args.out_dir):
        os.makedirs(args.out_dir)

    result_files = glob.glob(path.join(args.in_dir, "*.h5"))
    prog = tqdm(result_files)
    for file in prog:
        basename = path.basename(file)
        ds_name, ext = basename.split(".")
        prog.set_postfix_str(ds_name)

        res_obj = SuiteResult.load_hdf(file)
        k_a = {}
        for activation_fn in ("linear", "softmax"):
            for masker in ("constant", "blur", "random"):
                variant_name = f"{activation_fn} - {masker}"
                k_a[variant_name] = {}
                dfs = get_default_dfs(res_obj, mode="single", activation_fn=activation_fn, masker=masker)
                for metric_name, (df, inverted) in dfs.items():
                    if metric_name != "impact_coverage":
                        k_a[variant_name][metric_name] = krippendorff_alpha(df.to_numpy())
        k_a = pd.DataFrame(k_a)

        color_list = [
            "#004c6d",
            "#6996b3",
            "#c1e7ff",
            "#6d012a",
            "#b65a73",
            "#ffaac4"
        ]

        fig, ax = plt.subplots(figsize=(16, 8))
        k_a.plot.bar(ax=ax, width=0.7, color=color_list)
        plt.xticks(rotation=45, ha="right", rotation_mode="anchor")
        plt.grid(axis="x")
        fig.savefig(path.join(args.out_dir, f"{ds_name}.png"), bbox_inches="tight")
        plt.close(fig)
