import os
import numpy as np
import torch
import argparse
from util.tabular import _DATASETS, OpenMLDataset, BasicNN, get_method_dict
from attribench.data import AttributionsDataset, HDF5Dataset
from attribench.masking import TabularMasker
from attribench.functional.metrics.infidelity import (
    NoisyBaselinePerturbationGenerator,
    GaussianPerturbationGenerator,
)
from attribench.functional.metrics import (
    deletion,
    insertion,
    infidelity,
    sensitivity_n,
    minimal_subset,
    max_sensitivity,
)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, choices=_DATASETS.keys())
    parser.add_argument("--attrs-file", type=str)
    parser.add_argument("--samples-file", type=str)
    parser.add_argument("--data-dir", type=str)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--output-dir", type=str)
    parser.add_argument(
        "--metrics",
        nargs="*",
        default=[
            "deletion",
            "insertion",
            "infidelity",
            "sensitivity_n",
            "minimal_subset_deletion",
            "minimal_subset_insertion",
            "max_sensitivity",
        ],
    )
    args = parser.parse_args()

    # Get samples and attributions dataset
    samples_dataset = HDF5Dataset(args.samples_file)
    attrs_dataset = AttributionsDataset(
        samples_dataset, path=args.attrs_file
    )

    # Get train dataset
    ds_path = os.path.join(args.data_dir, args.dataset)
    ds_meta = _DATASETS[args.dataset]
    X_train = np.load(os.path.join(ds_path, "X_train.npy"))
    y_train = np.load(os.path.join(ds_path, "y_train.npy"))
    train_dataset = OpenMLDataset(X_train, y_train, ds_meta["pred_type"])
    num_inputs = X_train.shape[1]
    num_outputs = len(set(y_train))

    os.makedirs(args.output_dir, exist_ok=True)

    # Load model
    model = BasicNN(input_size=num_inputs, output_size=num_outputs)
    model_path = os.path.join(ds_path, "model.pt")
    model.load_state_dict(torch.load(model_path))

    # Data is normalized around 0, so masking with 0 is
    # the same as masking with average
    maskers = {"tabular": TabularMasker(mask_value=0)}

    method_dict = get_method_dict(model, train_dataset)
    activation_fns = ["linear", "softmax"]

    ############
    # DELETION #
    ############
    if "deletion" in args.metrics:
        for mode in ["morf", "lerf"]:
            print(f"Running Deletion-{mode}...")
            result = deletion(
                model,
                attrs_dataset,
                args.batch_size,
                maskers,
                activation_fns,
                mode,
                start=0.0,
                stop=0.5,
                num_steps=num_inputs // 2,
            )
            result.save(
                os.path.join(args.output_dir, f"deletion-{mode}"),
                format="csv",
            )

    #############
    # INSERTION #
    #############
    if "insertion" in args.metrics:
        for mode in ["morf", "lerf"]:
            print(f"Running Insertion-{mode}...")
            result = insertion(
                model,
                attrs_dataset,
                args.batch_size,
                maskers,
                activation_fns,
                mode,
                start=0.0,
                stop=0.5,
                num_steps=num_inputs // 2,
            )
            result.save(
                os.path.join(args.output_dir, f"insertion-{mode}"),
                format="csv",
            )

    ##############
    # INFIDELITY #
    ##############
    if "infidelity" in args.metrics:
        print("Running Infidelity...")
        perturbation_generators = {
            "noisy_baseline": NoisyBaselinePerturbationGenerator(sd=0.2),
            "gaussian": GaussianPerturbationGenerator(sd=0.2),
        }
        result = infidelity(
            model,
            attrs_dataset,
            args.batch_size,
            activation_fns,
            perturbation_generators,
            num_perturbations=1000,
        )
        result.save(os.path.join(args.output_dir, "infidelity"), format="csv")

    #################
    # SENSITIVITY_N #
    #################
    if "sensitivity_n" in args.metrics:
        print("Running Sensitivity-N...")
        result = sensitivity_n(
            model,
            attrs_dataset,
            args.batch_size,
            maskers,
            activation_fns,
            min_subset_size=0.1,
            max_subset_size=0.5,
            num_steps=10,
            num_subsets=100,
            segmented=False
        )
        result.save(
            os.path.join(args.output_dir, "sensitivity_n"), format="csv"
        )

    ###########################
    # MINIMAL_SUBSET_DELETION #
    ###########################
    if "minimal_subset_deletion" in args.metrics:
        print("Running Minimal Subset Deletion...")
        result = minimal_subset(
            model,
            attrs_dataset,
            args.batch_size,
            maskers,
            mode="deletion",
            num_steps=num_inputs,
        )
        result.save(
            os.path.join(args.output_dir, "minimal_subset_deletion"),
            format="csv",
        )

    ############################
    # MINIMAL_SUBSET_INSERTION #
    ############################
    if "minimal_subset_insertion" in args.metrics:
        print("Running Minimal Subset Insertion...")
        result = minimal_subset(
            model,
            attrs_dataset,
            args.batch_size,
            maskers,
            mode="insertion",
            num_steps=num_inputs,
        )
        result.save(
            os.path.join(args.output_dir, "minimal_subset_insertion"),
            format="csv",
        )

    ###################
    # MAX_SENSITIVITY #
    ###################
    if "max_sensitivity" in args.metrics:
        print("Running Max Sensitivity...")
        result = max_sensitivity(
            attrs_dataset,
            args.batch_size,
            method_dict,
            num_perturbations=50,
            radius=0.1
        )
        result.save(
            os.path.join(args.output_dir, "max_sensitivity"),
            format="csv",
        )
