import pandas as pd
from scipy.stats import ttest_rel
from attrbench.suite import SuiteResult
import numpy as np
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset", type=str, default="imagenet")
    args = parser.parse_args()

    ds_name = args.dataset

    variance_df = pd.read_csv(f"out/sens_n_variance/{ds_name}.csv")
    statistic, pvalue = ttest_rel(variance_df["sens_n"], variance_df["seg_sens_n"])
    print(f"{pvalue:.2E} " + ("SIG" if pvalue < 0.05 else "NOT SIG"))

    res = SuiteResult.load_hdf(f"out/{ds_name}.h5")
    sens_n_res = res.metric_results["sensitivity_n"].get_df()[0]["DeepShap"]
    seg_sens_n_res = res.metric_results["seg_sensitivity_n"].get_df()[0]["DeepShap"]

    print(f"Sensitivity-n noise fraction of variance: {variance_df['sens_n'].mean()/sens_n_res.var()*100:.2f}%")
    print(f"Seg-Sensitivity-n noise fraction of variance: {variance_df['seg_sens_n'].mean()/seg_sens_n_res.var()*100:.2f}%")
