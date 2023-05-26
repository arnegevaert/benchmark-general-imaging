from attrbench.suite.plot import InterMetricCorrelationPlot
from util.dfs import get_default_dfs, get_all_dfs


def inter_metric_correlations(res_obj, metric_selection="default"):
    if metric_selection == all:
        dfs = get_all_dfs(res_obj, mode="raw")
    else:
        dfs = get_default_dfs(res_obj, mode="raw")

    figsize = (17, 17) if metric_selection == "all" else (7, 7)
    fontsize = 15 if metric_selection == "all" else 17
    fig = InterMetricCorrelationPlot(dfs).render(
        figsize=figsize, annot=metric_selection == "all"
    )
    ax = fig.axes[0]
    ax.set_xticklabels(
        ax.get_xticklabels(),
        rotation=45,
        ha="right",
        rotation_mode="anchor",
        fontsize=fontsize,
    )
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=fontsize)
    return fig
