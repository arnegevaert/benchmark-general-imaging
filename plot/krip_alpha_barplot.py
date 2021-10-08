import argparse
import pandas as pd
from attrbench.suite import SuiteResult
from krip import compute_krippendorff_alpha, plot_krippendorff_alpha
import matplotlib as mpl
import numpy as np
from os import path
import seaborn as sns
from tqdm import tqdm


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("in_dir", type=str)
    parser.add_argument("out_file", type=str)
    parser.add_argument("--all", action="store_true")
    args = parser.parse_args()

    mpl.use("Agg")
    np.seterr(all="raise")
    sns.set()

    datasets = ["mnist", "fashionmnist", "cifar10", "cifar100", "svhn", "imagenet", "caltech", "places"]
    color = ["#036d08", "#9de78c", "#08036d", "#845eb3", "#e7c3ff", "#6d012a", "#b65a73", "#ffaac4"]

    result_objects = {ds: SuiteResult.load_hdf(path.join(args.in_dir, f"{ds}.h5")) for ds in datasets}
    k_a = {
        ds_name: compute_krippendorff_alpha(res, metric_selection="all" if args.all else "default")
        for ds_name, res in tqdm(result_objects.items())
    }
    k_a = pd.DataFrame(k_a)

    fig = plot_krippendorff_alpha(k_a, color)
    fig.savefig(args.out_file, bbox_inches="tight", dpi=250)
