import argparse
import pandas as pd
import os
from attribench.data import HDF5Dataset, AttributionsDataset
from attribench.distributed.metrics import SensitivityN
from attribench.masking.image import ConstantImageMasker
from util.datasets import ALL_DATASETS
from util.models import ModelFactoryImpl


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, choices=ALL_DATASETS)
    parser.add_argument("--model", type=str)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--iterations", type=int)
    parser.add_argument("--samples-file", type=str)
    parser.add_argument("--attrs-file", type=str)
    parser.add_argument("--data-dir", type=str)
    parser.add_argument("--out-dir", type=str)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    # Get dataset and attributions
    samples_dataset = HDF5Dataset(args.samples_file)
    attrs_dataset = AttributionsDataset(
        samples_dataset,
        path=args.attrs_file,
        aggregate_dim=0,
        aggregate_method="mean",
    )

    # Get model
    model_factory = ModelFactoryImpl(args.dataset, args.data_dir, args.model)

    # Check if output directory exists and is empty
    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)
    elif len(os.listdir(args.out_dir)) > 0 and not args.overwrite:
        raise ValueError("Output directory is not empty")
    
    sens_n_dfs = []
    seg_sens_n_dfs = []
    for i in range(args.iterations):
        print(f"Iteration {i+1}/{args.iterations}")
        sens_n = SensitivityN(
            model_factory=model_factory,
            attributions_dataset=attrs_dataset,
            batch_size=args.batch_size,
            maskers={"constant": ConstantImageMasker(masking_level="pixel")},
            activation_fns=["linear"],
            min_subset_size=0.1,
            max_subset_size=0.5,
            num_steps=10,
            num_subsets=100,
            segmented=False,
        )
        sens_n.run()
        result = sens_n.result
        df = result.get_df(masker="constant", activation_fn="linear")[0]
        sens_n_dfs.append(df)

        seg_sens_n = SensitivityN(
            model_factory=model_factory,
            attributions_dataset=attrs_dataset,
            batch_size=args.batch_size,
            maskers={"constant": ConstantImageMasker(masking_level="pixel")},
            activation_fns=["linear"],
            min_subset_size=0.1,
            max_subset_size=0.5,
            num_steps=10,
            num_subsets=100,
            segmented=True,
        )
        seg_sens_n.run()
        result = seg_sens_n.result
        df = result.get_df(masker="constant", activation_fn="linear")[0]
        seg_sens_n_dfs.append(df)

    sens_n_df = pd.concat(sens_n_dfs)
    seg_sens_n_df = pd.concat(seg_sens_n_dfs)

    sens_n_df.to_csv(
        os.path.join(args.out_dir, "sens_n.csv"), index_label="Sample"
    )
    seg_sens_n_df.to_csv(
        os.path.join(args.out_dir, "seg_sens_n.csv"), index_label="Sample"
    )
