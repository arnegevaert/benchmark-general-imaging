import argparse
from attribench.result import MetricResult
import os
from matplotlib import pyplot as plt
import numpy as np
import seaborn as sns

    
method_order = [
    "DeepSHAP",
    "ExpectedGradients",
    "DeepLIFT",
    "GradCAM",
    "GradCAM++",
    "ScoreCAM",
    "XRAI",
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

    metric_result = MetricResult.load(
        os.path.join(args.in_dir, "parameter_randomization.h5")
    )
    df, _ = metric_result.get_df()

    df.rename(
        columns={"DeepShap": "DeepSHAP", "DeepLift": "DeepLIFT"},
        inplace=True,
    )
    result = df.mean()
    result = result[method_order].abs()
    threshold = 0.05

    color_arr = np.where(result > threshold, "r", "g")
        
    sns.set_theme()
    fig, ax = plt.subplots(figsize=(7, 5))
    result.plot.bar(ax=ax, color=list(color_arr))
    ax.axhline(y=threshold, ls=':')

    fig.savefig(
        os.path.join(args.out_dir, "parameter_randomization.svg"),
        bbox_inches="tight",
    )