import argparse
import matplotlib as mpl
from matplotlib import pyplot as plt
import numpy as np
import os
from attribench.plot import WilcoxonSummaryPlot
from util.get_dataframes import get_dataframes
from tqdm import tqdm


METHOD_ORDER = [
    "DeepSHAP",
    "ExpectedGradients",
    "DeepLIFT",
    "GradCAM",
    "KernelSHAP",
    "LIME",
    "SmoothGrad",
    "VarGrad",
    "IntegratedGradients",
    "InputXGradient",
    "Gradient",
    "GuidedBackprop",
    "GuidedGradCAM",
    "Deconvolution",
]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--in-dir", type=str, default="out/results")
    parser.add_argument("-o", "--out-dir", type=str, default="out/plots")
    args = parser.parse_args()

    mpl.use("Agg")
    np.seterr(all="raise")

    datasets = os.listdir(args.in_dir)
    for ds_name in tqdm(datasets):
        in_dir = os.path.join(args.in_dir, ds_name)
        out_dir = os.path.join(args.out_dir, ds_name)
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

        for metric_selection in ("default", "all"):
            dfs = get_dataframes(in_dir, metric_selection, baseline="Random")
            fig = WilcoxonSummaryPlot(dfs).render(
                figsize=(10, 10)
                if metric_selection == "default"
                else (10, 25),
                glyph_scale=1000,
                fontsize=25,
                method_order=METHOD_ORDER,
            )
            fig.savefig(
                os.path.join(out_dir, f"wilcoxon_{metric_selection}.svg"),
                bbox_inches="tight",
            )
            plt.close(fig)
