import argparse
from attrbench.suite import SuiteResult
import matplotlib as mpl
import numpy as np
import seaborn as sns
from krip import compute_krippendorff_alpha, plot_krippendorff_alpha


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("in_file", type=str)
    parser.add_argument("out_file", type=str)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--threshold", type=float)
    args = parser.parse_args()

    mpl.use("Agg")
    np.seterr(all="raise")
    sns.set()

    result_object = SuiteResult.load_hdf(args.in_file)
    k_a = compute_krippendorff_alpha(result_object, metric_selection="all" if args.all else "default")
    k_a = k_a.to_frame("alpha")
    fig = plot_krippendorff_alpha(k_a)
    fig.savefig(args.out_file, bbox_inches="tight", dpi=250)

