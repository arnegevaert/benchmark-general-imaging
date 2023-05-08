import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


if __name__ == "__main__":
    data = 0.05 + np.random.rand(25) * 0.5

    sns.set()
    fig, ax = plt.subplots()
    ax.bar(x=np.arange(data.shape[0]), height=data)
    ax.set_xticks([])
    ax.set_yticks([])

    fig.savefig("bar.svg", bbox_inches="tight")