from dfs import get_all_dfs
from pingouin import wilcoxon
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt


def pairwise_tests(res_obj, method1, method2, title=None):
    dfs = get_all_dfs(res_obj, "single")

    result_cles = {}
    for key in dfs:
        df, inverted = dfs[key]
        df = -df if inverted else df

        res = wilcoxon(x=df[method1], y=df[method2], alternative="two-sided")
        pvalue = res["p-val"]["Wilcoxon"]
        cles = res["CLES"]["Wilcoxon"]
        out_str = f"{key}: p={pvalue:.3f} CLES={cles:.3f}"
        out_str += " *" if pvalue < 0.01 else " -"
        result_cles[key] = cles if pvalue < 0.01 else 0.5
    sns.set_color_codes("muted")
    df = pd.DataFrame(result_cles, index=["CLES"]).transpose().reset_index().rename(columns={"index": "Metric"})
    df["CLES"] -= 0.5
    fig, ax = plt.subplots(figsize=(5, 7))
    sns.barplot(data=df, x="CLES", y="Metric", color="b", left=0.5, ax=ax)
    ax.set(xlim=(0, 1))
    if title:
        ax.set_title(title)
    return fig
