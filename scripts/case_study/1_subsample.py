"""
In this script, we sample results for a randomly selected subset of samples.
In this way, we can simulate having run the full benchmark on a
smaller subset before running a selection of metrics on the full
dataset.
"""
import os
import argparse
import numpy as np
from numpy import typing as npt
import h5py
from tqdm import tqdm


def subsample_file(
    in_filename: str, out_filename: str, sample_indices: npt.NDArray
):
    def _subsample_rec(in_node: h5py.Group, out_node: h5py.Group):
        for key in in_node.keys():
            next_node = in_node[key]
            if isinstance(next_node, h5py.Dataset):
                # Copy the selected samples from the dataset
                out_node.create_dataset(
                    key,
                    data=next_node[sample_indices],
                )
                for attr in next_node.attrs:
                    out_node[key].attrs[attr] = next_node.attrs[attr]
            elif isinstance(next_node, h5py.Group):
                # Proceed recursively
                _subsample_rec(next_node, out_node.create_group(key))
        for attr in in_node.attrs:
            out_node.attrs[attr] = in_node.attrs[attr]

    _subsample_rec(h5py.File(in_filename), h5py.File(out_filename, "w"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("in_dir", type=str)
    parser.add_argument("out_dir", type=str)
    parser.add_argument("--num-samples", type=int, required=True)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    with h5py.File(os.path.join(args.in_dir, "samples.h5")) as samples_fp:
        samples_ds: h5py.Dataset = samples_fp["samples"]  # type: ignore
        sample_indices = np.random.choice(
            np.arange(samples_ds.shape[0]),
            size=args.num_samples,
            replace=False,
        )
        sample_indices.sort()
    
    for filename in tqdm(os.listdir(args.in_dir)):
        in_path = os.path.join(args.in_dir, filename)
        out_path = os.path.join(args.out_dir, filename)
        subsample_file(in_path, out_path, sample_indices)