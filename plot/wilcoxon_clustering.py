import argparse
from tqdm import tqdm
import os
from attrbench.suite import SuiteResult
from attrbench.lib.stat import wilcoxon_tests
from util.dfs import get_default_dfs
import pandas as pd
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("in_dir", type=str)
    parser.add_argument("out_dir", type=str)
    args = parser.parse_args()

    mpl.use("Agg")

    datasets = ["mnist", "fashionmnist", "cifar10", "cifar100", "svhn", "imagenet", "caltech", "places"]
    result_objects = {
        os.path.basename(filename).split(".")[0]: SuiteResult.load_hdf(filename)
        for filename in [os.path.join(args.in_dir, f"{ds}.h5") for ds in datasets]
    }

    for ds_name, res_obj in tqdm(result_objects.items()):
        dfs = get_default_dfs(res_obj, mode="single")

        pvalues, effect_sizes = {}, {}
        for metric_name, (df, inverted) in dfs.items():
            mes, mpv = wilcoxon_tests(df, inverted)
            effect_sizes[metric_name] = mes
            pvalues[metric_name] = mpv
        pvalues = pd.DataFrame(pvalues)
        effect_sizes = pd.DataFrame(effect_sizes).abs()
        effect_sizes[pvalues > 0.01] = 0
        
        effect_sizes = effect_sizes / effect_sizes.max()
        effect_sizes = effect_sizes.fillna(0)

        fig = sns.clustermap(effect_sizes.transpose(), row_cluster=False, cmap="Greens", figsize=(7,7), method="single", metric="correlation")
        fig.savefig(os.path.join(args.out_dir, f"{ds_name}.png"), bbox_inches="tight", dpi=250)