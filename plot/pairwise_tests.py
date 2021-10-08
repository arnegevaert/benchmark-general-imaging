import argparse
from attrbench.suite import SuiteResult
from dfs import get_all_dfs
from pingouin import wilcoxon
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("hdf_file", type=str)
    parser.add_argument("out_file")
    args = parser.parse_args()
    mpl.use("Agg")
    sns.set_style("whitegrid")

    res_obj = SuiteResult.load_hdf(args.hdf_file)
    dfs = get_all_dfs(res_obj, "single")

    result_cles = {}
    for key in dfs:
        df, inverted = dfs[key]
        df = -df if inverted else df

        res = wilcoxon(x=df["DS"], y=df["DL"], tail="two-sided")
        pvalue = res["p-val"]["Wilcoxon"]
        cles = res["CLES"]["Wilcoxon"]
        out_str = f"{key}: p={pvalue:.3f} CLES={cles:.3f}"
        out_str += " *" if pvalue < 0.01 else " -"
        # print(out_str)
        result_cles[key] = cles if pvalue < 0.01 else 0.5

    sns.set_color_codes("muted")
    df = pd.DataFrame(result_cles, index=["CLES"]).transpose().reset_index().rename(columns={"index": "Metric"})
    df["CLES"] -= 0.5
    fig, ax = plt.subplots(figsize=(5, 7))
    sns.barplot(data=df, x="CLES", y="Metric", color="b", left=0.5, ax=ax)
    #ax.set_xticklabels(ax.get_xticklabels(), size=10)
    #ax.set_yticklabels(ax.get_yticklabels(), size=15)
    ax.set(xlim=(0, 1))
    ax.set_title(f"P(DeepSHAP > DeepLIFT)")
    fig.savefig(args.out_file, bbox_inches="tight", dpi=250)
    plt.close(fig)
