import argparse
from util.models import ModelFactoryImpl
from util.datasets import ALL_DATASETS, get_dataset
from attribench.data import HDF5Dataset
from attribench.distributed import ComputeAttributions
from util.attribution.method_factory import get_method_factory


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        choices=ALL_DATASETS,
        help="Dataset to use",
        required=True,
    )
    parser.add_argument(
        "--samples-file",
        type=str,
        help="HDF5 file with samples (produced by 1_select_samples.py)",
        required=True,
    )
    parser.add_argument(
        "--model", type=str, help="Name of model to use", required=True
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        help="Path to data directory containing datasets and models",
        required=True,
    )
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument(
        "--output-file", type=str, help="Path to store results", required=True
    )
    parser.add_argument(
        "--methods", type=str, nargs="*", help="Methods to use (default: all)"
    )
    args = parser.parse_args()

    dataset = HDF5Dataset(args.samples_file)
    reference_dataset = get_dataset(args.dataset, args.data_dir)
    model_factory = ModelFactoryImpl(args.dataset, args.data_dir, args.model)

    method_factory = get_method_factory(
        args.batch_size,
        reference_dataset=reference_dataset,
        methods=args.methods,
    )

    computation = ComputeAttributions(
        model_factory,
        method_factory,
        dataset,
        batch_size=args.batch_size,
    )
    computation.run(args.output_file)
