import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt


if __name__ == "__main__":
    data = np.random.rand(10, 10)
    data = (data + data.T)/2
    data = (data - data.min()) / (data.max() - data.min()) * 2 - 1
    fig, ax = plt.subplots()
    sns.heatmap(data, vmin=-1, vmax=1, cmap=sns.diverging_palette(220, 20, as_cmap=True), cbar=False,
                xticklabels=False, yticklabels=False)
    ax.set_aspect("equal")
    fig.savefig("corr.svg", bbox_inches="tight")
