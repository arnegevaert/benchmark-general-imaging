import numpy as np
import torch
from util.tabular import _DATASETS, OpenMLDataset, BasicNN
from attribench.data import HDF5DatasetWriter
import os
import argparse
from attribench.functional import select_samples


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, choices=_DATASETS.keys())
    parser.add_argument("--data-dir", type=str)
    parser.add_argument("--output-file", type=str)
    parser.add_argument("--num-samples", type=int)
    args = parser.parse_args()

    # Get test dataset
    ds_path = os.path.join(args.data_dir, args.dataset)
    ds_meta = _DATASETS[args.dataset]
    X_test = np.load(os.path.join(ds_path, "X_test.npy"))
    y_test = np.load(os.path.join(ds_path, "y_test.npy"))
    test_dataset = OpenMLDataset(X_test, y_test, ds_meta["pred_type"])

    # Load model
    model = BasicNN(input_size=X_test.shape[1], output_size=len(set(y_test)))
    model_path = os.path.join(ds_path, "model.pt")
    model.load_state_dict(torch.load(model_path))

    # Select correctly classified samples
    writer = HDF5DatasetWriter(args.output_file, args.num_samples)
    select_samples(
        model,
        test_dataset,
        num_samples=args.num_samples,
        batch_size=32,
        writer=writer,
    )
