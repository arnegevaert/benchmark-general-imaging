import argparse
import pandas as pd
from attrbench.lib import krippendorff_alpha
from attrbench.suite import SuiteResult
from dfs import get_default_dfs, get_all_dfs
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from os import path
import glob
import seaborn as sns
from tqdm import tqdm


def compute_krippendorff_alpha(result_object, metric_selection="default"):
    if metric_selection == "all":
        dfs = get_all_dfs(result_object, mode="single").items()
    else:
        dfs = get_default_dfs(result_object, mode="single").items()
    k_a = {
        metric_name: krippendorff_alpha(df.to_numpy()) for metric_name, (df, _) in dfs
    }
    return pd.Series(k_a)


def plot_krippendorff_alpha(k_a: pd.DataFrame, color=None):
    num_rows = k_a.shape[0] // 12 + 1
    fig, axs = plt.subplots(nrows=num_rows, figsize=(10, 4*num_rows), constrained_layout=True)

    for i in range(num_rows):
        end = min((i+1)*12, k_a.shape[0])
        k_a[i*12:end].plot.bar(ax=axs[i], color=color, width=0.7)
        axs[i].set_xticklabels(axs[i].get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        if i > 0:
            axs[i].legend().set_visible(False)
        else:
            axs[i].legend(loc='upper center', bbox_to_anchor=(0.5, 1.4),
                          fancybox=True, shadow=True, ncol=4)
        plt.grid(axis="x")

    return fig
