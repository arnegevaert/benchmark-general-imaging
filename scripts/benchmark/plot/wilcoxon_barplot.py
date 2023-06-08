import argparse
import matplotlib as mpl
from matplotlib import pyplot as plt
import numpy as np
import os
from attrbench.plot import WilcoxonBarPlot
from util.get_dataframes import get_dataframes


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
    parser.add_argument("-i", "--in-dir", type=str, default="out/results/MNIST")
    parser.add_argument("-o", "--out-dir", type=str, default="out/plots/MNIST")
    args = parser.parse_args()

    mpl.use("Agg")
    np.seterr(all="raise")

    if not os.path.isdir(os.path.join(args.out_dir, "wilcoxon_bar")):
        os.makedirs(os.path.join(args.out_dir, "wilcoxon_bar"))

    for metric_selection in ("default", "all"):
        dfs = get_dataframes(args.in_dir, metric_selection, baseline="Random")
        fig = WilcoxonBarPlot(dfs).render()
        fig.savefig(
            os.path.join(
                args.out_dir, "wilcoxon_bar", f"wilcoxon_{metric_selection}.svg"
            ),
            bbox_inches="tight",
        )
        plt.close(fig)
