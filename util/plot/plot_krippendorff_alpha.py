import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def plot_krippendorff_alpha(
    k_a: pd.DataFrame,
    color=None,
    color_thresh=None,
    groups_per_row=14,
    legend=True,
):
    if color is None and color_thresh is None:
        raise ValueError("Specify color or color_thresh")

    num_rows = (k_a.shape[0] - 1) // groups_per_row + 1
    fig, axs = plt.subplots(
        nrows=num_rows, figsize=(10, 4 * num_rows), constrained_layout=True
    )

    if num_rows == 1:
        axs = [axs]

    for i in range(num_rows):
        end = min((i + 1) * groups_per_row, k_a.shape[0])
        part = k_a[i * groups_per_row : end]

        if color is None:
            color_arr = np.where(part["alpha"] > color_thresh, "g", "r")
            part.plot.bar(
                y="alpha",
                use_index=True,
                ax=axs[i],
                color=color_arr,
                width=0.7,
            )
        else:
            part.plot.bar(ax=axs[i], color=color, width=0.7)

        axs[i].set_xticklabels(
            axs[i].get_xticklabels(),
            rotation=45,
            ha="right",
            rotation_mode="anchor",
        )

        if i > 0 or not legend:
            axs[i].legend().set_visible(False)
        else:
            axs[i].legend(
                loc="upper center",
                bbox_to_anchor=(0.5, 1.4),
                fancybox=True,
                shadow=True,
                ncol=4,
            )
        plt.grid(axis="x")
    return fig
