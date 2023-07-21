import argparse
import matplotlib as mpl
from matplotlib import pyplot as plt
import numpy as np
import os
from attribench.result import MetricResult
from attribench.plot import InterMetricCorrelationPlot


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--in-dir", type=str)
    parser.add_argument("-o", "--out-dir", type=str)
    args = parser.parse_args()

    mpl.use("Agg")
    np.seterr(all="raise")

    out_dir = os.path.join(args.out_dir, "masking_correlations")
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    metric_names = [
        "deletion_morf",
        "deletion_lerf",
        "insertion_morf",
        "insertion_lerf",
        "irof_morf",
        "irof_lerf",
        "minimal_subset_deletion",
        "minimal_subset_insertion",
    ]
    available_files = os.listdir(args.in_dir)

    for metric_name in metric_names:
        if metric_name + ".h5" in available_files:
            result_object = MetricResult.load(
                os.path.join(args.in_dir, metric_name + ".h5")
            )
            dfs = {}
            for masker in ["constant", "blurring", "random"]:
                dfs[masker] = result_object.get_df(
                    masker=masker, activation_fn="linear"
                )
            fig = InterMetricCorrelationPlot(dfs).render(
                figsize=(3, 3), annot=True, fontsize=15
            )

            ax = fig.axes[0]
            ax.set_yticklabels(ax.get_yticklabels(), rotation=0)

            fig.savefig(
                os.path.join(
                    out_dir,
                    f"{metric_name}.svg",
                ),
                bbox_inches="tight",
            )
            plt.close(fig)
