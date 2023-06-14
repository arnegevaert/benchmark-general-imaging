import argparse
import pandas as pd
from tqdm import trange
import os
from attribench.data import HDF5Dataset, AttributionsDataset
from attribench.metrics import SensitivityN
from attribench.masking import ConstantMasker
from util.datasets import ALL_DATASETS
from util.models import ModelFactoryImpl


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset", type=str, default="MNIST", choices=ALL_DATASETS
    )
    parser.add_argument("--model", type=str, default="CNN")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--samples-file", type=str, default="samples.h5")
    parser.add_argument("--attrs-file", type=str, default="attributions.h5")
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--out-dir", type=str, default="out")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    # Get dataset and attributions
    samples_dataset = HDF5Dataset(args.samples_file)
    attrs_dataset = AttributionsDataset(
        samples_dataset,
        args.attrs_file,
        aggregate_axis=0,
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
    for _ in trange(args.iterations):
        for segmented in [True, False]:
            # Run Sensitivity-n on the full dataset
            sens_n = SensitivityN(
                model_factory,
                attrs_dataset,
                args.batch_size,
                min_subset_size=0.1,
                max_subset_size=0.5,
                num_steps=10,
                num_subsets=100,
                maskers={"constant": ConstantMasker(feature_level="pixel")},
                activation_fns=["linear"],
                segmented=segmented,
            )
            sens_n.run()

            df = sens_n.result.get_df(
                masker="constant", activation_fn="linear"
            )[0]
            if segmented:
                seg_sens_n_dfs.append(df)
            else:
                sens_n_dfs.append(df)

    sens_n_df = pd.concat(sens_n_dfs)
    seg_sens_n_df = pd.concat(seg_sens_n_dfs)

    sens_n_df.to_csv(
        os.path.join(args.out_dir, "sens_n.csv"), index_label="Sample"
    )
    seg_sens_n_df.to_csv(
        os.path.join(args.out_dir, "seg_sens_n.csv"), index_label="Sample"
    )
