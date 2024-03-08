import argparse
import os
from scipy import stats
from util.get_dataframes import get_dataframes
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt


def _generate_plot(method1, method2, dfs):
    result_cles = {}
    for key in dfs:
        df, higher_is_better = dfs_filtered[key]
        if not higher_is_better:
            df = -df

        _, pvalue = stats.ttest_1samp(df[method1] - df[method2], 0, alternative="two-sided")
        cles = (df[method1] > df[method2]).mean()
        result_cles[key] = cles if pvalue < 0.01 else 0.5
    sns.set_theme()
    df = (
        pd.DataFrame(result_cles, index=["CLES"])
        .transpose()
        .reset_index()
        .rename(columns={"index": "Metric"})
    )
    df["CLES"] -= 0.5
    fig, ax = plt.subplots(figsize=(4, 4))
    sns.barplot(data=df, x="CLES", y="Metric", color="b", left=0.5, ax=ax)
    ax.set(xlim=(0, 1))
    ax.set_title(f"P({method1} > {method2})")
    fig.savefig(
        os.path.join(args.out_dir, f"cles_{method1}_{method2}.svg"),
        bbox_inches="tight",
    )
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("in_dir", type=str)
    parser.add_argument("out_dir", type=str)
    args = parser.parse_args()

    combinations = [
        ("DeepSHAP", "DeepLIFT"),
        ("KernelSHAP", "GradCAM"),
        ("DeepSHAP", "GradCAM"),
    ]
    
    metric_order = [
        "Cov",
        "MaxSens",
        "Ins - MoRF - random",
        "IROF - MoRF - constant",
        "IROF - LeRF - random",
        "SegSensN - constant",
        "INFD - SQ",
        "INFD - BL",
        "MSIns - random",
        "MSDel - constant",
    ]
    dfs = get_dataframes(
        args.in_dir, "all", baseline="Random", data_type="image"
    )
    dfs_filtered = {key: dfs[key] for key in metric_order}


    for method1, method2 in combinations:
        _generate_plot(method1, method2, dfs_filtered)