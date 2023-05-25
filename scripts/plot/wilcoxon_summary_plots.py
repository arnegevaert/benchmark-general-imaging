import argparse
import matplotlib as mpl
from matplotlib import pyplot as plt
import numpy as np
import os
from attrbench.plot import WilcoxonSummaryPlot
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
    parser.add_argument("in_dir", type=str)
    parser.add_argument("out_dir", type=str)
    args = parser.parse_args()

    mpl.use("Agg")
    np.seterr(all="raise")

    if not os.path.isdir(os.path.join(args.out_dir, "wilcoxon")):
        os.makedirs(os.path.join(args.out_dir, "wilcoxon"))

    for metric_selection in ("default", "all"):
        dfs = get_dataframes(args.in_dir, metric_selection)
        fig = WilcoxonSummaryPlot(dfs).render(
            figsize=(10, 10) if metric_selection == "default" else (10, 25),
            glyph_scale=1000,
            fontsize=25,
            method_order=METHOD_ORDER,
        )
        fig.savefig(
            os.path.join(
                args.out_dir, "wilcoxon", f"wilcoxon_{metric_selection}.svg"
            ),
            bbox_inches="tight",
        )
        plt.close(fig)
