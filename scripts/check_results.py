import argparse
import os
import h5py
import numpy as np


def check_nan_visitor(name, node):
    if isinstance(node, h5py.Dataset):
        data = node[:]
        if np.isnan(data).any():
            print(f"    Found NaNs in {name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Check all results, samples and attributions in a directory"
        " for NaNs and Infs."
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
        for file in os.listdir(os.path.join(args.dir, ds)):
            print(f"  Checking file {file}")
            with h5py.File(os.path.join(args.dir, ds, file), "r") as fp:
                fp.visititems(check_nan_visitor)
