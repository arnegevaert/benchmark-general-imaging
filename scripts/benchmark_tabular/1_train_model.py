import os
import argparse
from sklearn.datasets import fetch_openml
from torch.utils.data import DataLoader
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
from util.tabular import OpenMLDataset, BasicNN, _DATASETS


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, choices=_DATASETS.keys())
    parser.add_argument("--num-epochs", type=int, default=10)
    parser.add_argument("--data-dir", type=str)
    args = parser.parse_args()

    # Fetch the dataset
    ds_meta = _DATASETS[args.dataset]
    dataset = fetch_openml(data_id=ds_meta["data_id"], parser="auto")

    # Only select rows without nan
    target = dataset.target[dataset.data.notna().all(axis=1)]
    data = dataset.data[dataset.data.notna().all(axis=1)]

    # Construct train test split and save to disk
    X_train, X_test, y_train, y_test = train_test_split(
        data, target, test_size=0.2, random_state=42
    )

    # Print dataset info
    print(f"Dataset: {args.dataset}")
    print(f"Number of samples: {len(dataset.data)}")
    print(f"Number of features: {len(dataset.feature_names)}")
    print(f"Number of training samples: {len(X_train)}")
    print(f"Number of test samples: {len(X_test)}")
    if ds_meta["pred_type"] == "classification":
        print(f"Number of classes: {len(set(dataset.target))}")

    # If classification, convert to integer labels
    if ds_meta["pred_type"] == "classification":
        unique_labels = set(dataset.target)
        label_to_int = {label: i for i, label in enumerate(unique_labels)}
        y_train = y_train.map(label_to_int)
        y_test = y_test.map(label_to_int)

    # Integer encode categorical columns
    for col in X_train.columns:
        if dataset.data[col].dtype == "category":
            X_train[col] = X_train[col].cat.codes
            X_test[col] = X_test[col].cat.codes

    # Integer encode target if necessary
    if ds_meta["pred_type"] == "classification":
        if y_train.dtype == "category":
            y_train = y_train.cat.codes
            y_test = y_test.cat.codes
    
    # Standard normalize data
    for col in X_train.columns:
        scaler = StandardScaler()
        scaler.fit(X_train[col].values.reshape(-1, 1))
        X_train[col] = scaler.transform(X_train[col].values.reshape(-1, 1))
        X_test[col] = scaler.transform(X_test[col].values.reshape(-1, 1))

    train_dataset = OpenMLDataset(
        X_train.values, y_train.values, ds_meta["pred_type"]
    )
    test_dataset = OpenMLDataset(
        X_test.values, y_test.values, ds_meta["pred_type"]
    )

    # Construct model and train
    model = BasicNN(
        input_size=X_train.shape[1],
        output_size=len(set(y_train))
        if ds_meta["pred_type"] == "classification"
        else 1,
        pred_type=ds_meta["pred_type"],
    )
    train_dl = DataLoader(train_dataset, batch_size=32, shuffle=True)
    model.fit(train_dl, num_epochs=args.num_epochs)

    # Evaluate model on test set
    train_performance = model.test(DataLoader(train_dataset, batch_size=128))
    test_performance = model.test(DataLoader(test_dataset, batch_size=128))
    if ds_meta["pred_type"] == "classification":
        print(f"Train accuracy: {train_performance}")
        print(f"Test accuracy: {test_performance}")
    else:
        print(f"Train R2: {train_performance}")
        print(f"Test R2: {test_performance}")

    # Save model to disk
    ds_path = os.path.join(args.data_dir, args.dataset)
    os.makedirs(ds_path, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(ds_path, "model.pt"))

    # Save train and test data to disk
    np.save(os.path.join(ds_path, "X_train.npy"), X_train.values)
    np.save(os.path.join(ds_path, "X_test.npy"), X_test.values)
    np.save(os.path.join(ds_path, "y_train.npy"), y_train.values)
    np.save(os.path.join(ds_path, "y_test.npy"), y_test.values)