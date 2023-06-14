import argparse
from util.models import ModelFactoryImpl
from util.datasets import ALL_DATASETS, get_dataset
from attribench.data import AttributionsDatasetWriter, HDF5Dataset
from attribench.distributed import AttributionsComputation
from util.attribution.method_factory import get_method_factory


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset", type=str, default="MNIST", choices=ALL_DATASETS
    )
    parser.add_argument("--samples-file", type=str, default="samples.h5")
    parser.add_argument("--model", type=str, default="BasicCNN")
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--output-file", type=str, default="attributions.h5")
    args = parser.parse_args()

    dataset = HDF5Dataset(args.samples_file)
    reference_dataset = get_dataset(args.dataset, args.data_dir)
    writer = AttributionsDatasetWriter(
        args.output_file,
        num_samples=len(dataset),
        sample_shape=dataset.sample_shape,
    )
    model_factory = ModelFactoryImpl(args.dataset, args.data_dir, args.model)

    method_factory = get_method_factory(
        args.batch_size, reference_dataset=reference_dataset
    )

    computation = AttributionsComputation(
        model_factory,
        method_factory,
        dataset,
        batch_size=args.batch_size,
        writer=writer,
    )
    computation.run()
