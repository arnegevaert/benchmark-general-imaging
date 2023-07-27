from experiments.general_imaging.plot.dfs import get_all_dfs
from attribench.suite import SuiteResult
from pingouin import wilcoxon
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt


if __name__ == "__main__":
    res_obj = SuiteResult.load_hdf("../../out/imagenet.h5")
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
    ax.axes.xaxis.set_visible(False)
    ax.axes.yaxis.set_visible(False)
    fig.savefig("cles.svg", bbox_inches="tight")
