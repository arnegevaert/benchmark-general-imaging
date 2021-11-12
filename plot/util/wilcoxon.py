from attrbench.suite.plot import WilcoxonSummaryPlot
from util.dfs import get_default_dfs, get_all_dfs


def wilcoxon(res_obj, metric_selection="default"):
    if metric_selection == "all":
        dfs = get_all_dfs(res_obj, mode="single")
    else:
        dfs = get_default_dfs(res_obj, mode="single")
    figsize = (10, 25) if metric_selection == "all" else (10, 10)
    glyph_scale = 1000
    order = ["DeepShap", "ExpectedGradients", "DeepLift", "GradCAM", "KernelShap", "LIME", "SmoothGrad", "VarGrad",
             "IntegratedGradients", "InputXGradient", "Gradient", "GuidedBackprop", "GuidedGradCAM", "Deconvolution"]
    fig = WilcoxonSummaryPlot(dfs).render(figsize=figsize, glyph_scale=glyph_scale, fontsize=25, method_order=order)
    return fig
