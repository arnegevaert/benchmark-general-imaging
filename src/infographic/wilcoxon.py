from attrbench.suite import SuiteResult
from experiments.general_imaging.plot.dfs import get_default_dfs
from attrbench.suite.plot import WilcoxonSummaryPlot
import matplotlib.pyplot as plt

if __name__ == "__main__":
    res_obj = SuiteResult.load_hdf("../../out/svhn.h5")
    dfs = get_default_dfs(res_obj, mode="single")
    fig = WilcoxonSummaryPlot(dfs).render(figsize=(10, 10), glyph_scale=1000)
    ax = plt.gca()
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    fig.savefig("wilcoxon.svg", bbox_inches="tight")