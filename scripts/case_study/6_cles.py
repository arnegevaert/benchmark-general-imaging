import argparse
import os
from pingouin import wilcoxon
from util.get_dataframes import get_dataframes
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("in_dir", type=str)
    parser.add_argument("out_dir", type=str)
    args = parser.parse_args()

    method1 = "DeepSHAP"
    method2 = "GradCAM++"

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

    result_cles = {}
    for key in dfs_filtered:
        df, higher_is_better = dfs_filtered[key]
        if not higher_is_better:
            df = -df

        res = wilcoxon(x=df[method1], y=df[method2], alternative="two-sided")
        pvalue = res["p-val"]["Wilcoxon"]
        cles = res["CLES"]["Wilcoxon"]
        result_cles[key] = cles if pvalue < 0.01 else 0.5
    sns.set_color_codes("muted")
    df = (
        pd.DataFrame(result_cles, index=["CLES"])
        .transpose()
        .reset_index()
        .rename(columns={"index": "Metric"})
    )
    df["CLES"] -= 0.5
    fig, ax = plt.subplots(figsize=(5, 7))
    sns.barplot(data=df, x="CLES", y="Metric", color="b", left=0.5, ax=ax)
    ax.set(xlim=(0, 1))
    ax.set_title(f"P({method1} > {method2})")
    fig.savefig(
        os.path.join(args.out_dir, f"cles.svg"),
        bbox_inches="tight",
    )
    plt.close(fig)
