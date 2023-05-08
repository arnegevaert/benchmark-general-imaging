from models import get_model
from datasets import get_dataset, SAMPLE_SHAPES, ALL_DATASETS
from attrbench.distributed import Model
from attrbench.data import HDF5DatasetWriter
from attrbench.distributed import SampleSelection
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="MNIST",
                        choices=ALL_DATASETS)
    parser.add_argument("--model", type=str, default="BasicCNN")
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--num-samples", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--output-file", type=str, default="samples.h5")
    args = parser.parse_args()

    model = Model(get_model(args.dataset, args.data_dir, args.model))

    writer = HDF5DatasetWriter(
        path="samples.h5", num_samples=args.num_samples,
        sample_shape=SAMPLE_SHAPES[args.dataset])
    sample_selection = SampleSelection(
        model, get_dataset(args.dataset, args.data_dir),
         writer, num_samples=args.num_samples, batch_size=args.batch_size)
    sample_selection.run()
