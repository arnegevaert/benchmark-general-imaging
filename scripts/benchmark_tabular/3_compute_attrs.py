from util.tabular import _DATASETS, OpenMLDataset, BasicNN, get_method_dict
from attribench.data import HDF5Dataset
import numpy as np
import torch
import os
import argparse
from attribench.functional import compute_attributions
from attribench.data import AttributionsDatasetWriter


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, choices=_DATASETS.keys())
    parser.add_argument("--data-dir", type=str)
    parser.add_argument("--samples-file", type=str)
    parser.add_argument("--output-file", type=str)
    args = parser.parse_args()

    # Get samples
    dataset = HDF5Dataset(args.samples_file)

    # Get train and test dataset
    ds_path = os.path.join(args.data_dir, args.dataset)
    ds_meta = _DATASETS[args.dataset]
    X_train = np.load(os.path.join(ds_path, "X_train.npy"))
    y_train = np.load(os.path.join(ds_path, "y_train.npy"))
    train_dataset = OpenMLDataset(X_train, y_train, ds_meta["pred_type"])
    num_inputs = X_train.shape[1]
    num_outputs = len(set(y_train))

    # Load model
    model = BasicNN(input_size=num_inputs, output_size=num_outputs)
    model_path = os.path.join(ds_path, "model.pt")
    model.load_state_dict(torch.load(model_path))

    # Make method objects
    method_dict = get_method_dict(model, reference_dataset=train_dataset)

    # Compute attributions
    writer = AttributionsDatasetWriter(args.output_file, len(dataset))
    compute_attributions(
        model, method_dict, dataset, batch_size=32, writer=writer
    )
