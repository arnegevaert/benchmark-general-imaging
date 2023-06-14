from util.models import ModelFactoryImpl
from util.datasets import get_dataset, SAMPLE_SHAPES, ALL_DATASETS
from attribench.data import HDF5DatasetWriter
from attribench.distributed import SampleSelection
import argparse
import os


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset", type=str, default="MNIST", choices=ALL_DATASETS
    )
    parser.add_argument("--model", type=str, default="BasicCNN")
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--num-samples", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--output-file", type=str, default="samples.h5")
    parser.add_argument("--allow-overwrite", action="store_true")
    args = parser.parse_args()

    if os.path.exists(args.output_file):
        if args.allow_overwrite:
            os.remove(args.output_file)
        else:
            raise ValueError(
                "Output file exists. Pass --allow-overwrite to"
                " allow the script to overwrite this file."
            )

    # model = Model(get_model(args.dataset, args.data_dir, args.model))
    model_factory = ModelFactoryImpl(args.dataset, args.data_dir, args.model)

    writer = HDF5DatasetWriter(
        path=args.output_file,
        num_samples=args.num_samples,
        sample_shape=SAMPLE_SHAPES[args.dataset],
    )
    sample_selection = SampleSelection(
        model_factory,
        get_dataset(args.dataset, args.data_dir),
        writer,
        num_samples=args.num_samples,
        batch_size=args.batch_size,
    )
    sample_selection.run()
