import argparse
import os
import h5py
import numpy as np
import pandas as pd


def check_nan_h5(path):
    with h5py.File(path, "r") as fp:
        fp.visititems(check_nan_visitor)


def check_nan_visitor(name, node):
    if isinstance(node, h5py.Dataset):
        data = node[:]
        if np.isnan(data).any():
            print(f"    Found NaNs in {name}")
        if np.isinf(data).any():
            print(f"    Found Infs in {name}")


def check_nan_csv(path):
    def _check_nan_rec(_path):
        if os.path.isfile(_path) and _path.endswith(".csv"):
            data = np.loadtxt(_path)
            if np.isnan(data).any():
                print(f"    Found NaNs in {_path}")
            if np.isinf(data).any():
                print(f"    Found Infs in {_path}")
        elif os.path.isdir(_path):
            for subpath in os.listdir(_path):
                _check_nan_rec(os.path.join(_path, subpath))
    _check_nan_rec(path)


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
            path = os.path.join(args.dir, ds, file)
            if file.endswith(".h5"):
                check_nan_h5(path)
            else:
                check_nan_csv(path)
