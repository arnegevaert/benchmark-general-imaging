from scipy.stats import normaltest
import os
import argparse
from util.get_dataframes import get_dataframes


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Check all results, samples and attributions in a directory"
        " for normality."
    )
    parser.add_argument(
        "dir",
        type=str,
        help="Path to the directory containing the results to check.",
    )
    args = parser.parse_args()

    datasets = [
        filename
        for filename in os.listdir(args.dir)
        if os.path.isdir(os.path.join(args.dir, filename))
    ]
    for ds in datasets:
        print(f"Checking dataset {ds}")
        dfs = get_dataframes(os.path.join(args.dir, ds), mode="all")
        for key in dfs:
            normal_count, non_normal_count, nan_count = 0, 0, 0
            df, _ = dfs[key]
            for col in df.columns:
                try:
                    _, p = normaltest(df[col], nan_policy="raise")
                    if p < 0.05:
                        non_normal_count += 1
                    else:
                        normal_count += 1
                except ValueError:
                    nan_count += 1
            print(f"  {key}: Normal: {normal_count/len(df.columns)*100:.2f}%, "
                f"Non-normal: {non_normal_count/len(df.columns)*100:.2f}%, "
                f"NaNs: {nan_count/len(df.columns)*100:.2f}%"
            )
