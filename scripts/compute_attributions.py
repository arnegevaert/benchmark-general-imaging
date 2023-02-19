import argparse
from attrbench import MethodFactory
from attrbench.data import AttributionsDatasetWriter, HDF5Dataset
from captum import attr
from attribution import Deconvolution, DeepShap, DeepLift, EdgeDetection,\
        ExpectedGradients, GradCAM, Gradient


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--dataset", type=str, required=True)
    parser.add_argument("-b", "--batch-size", type=int, required=True)
    parser.add_argument("-o", "--output-file", type=str, required=True)
    args = parser.parse_args()

    dataset = HDF5Dataset(args.dataset)
    writer = AttributionsDatasetWriter(
            args.output_file, num_samples=len(dataset),
            sample_shape=dataset.sample_shape)

    method_factory = MethodFactory({
        "Deconvolution": Deconvolution,
        "DeepShap": (DeepShap, {"reference_dataset": dataset,
                                "n_baseline_samples": 100}),
        "DeepLift": DeepLift,
        "EdgeDetection": EdgeDetection,
        "ExpectedGradients": (ExpectedGradients, {"reference_dataset": dataset,
                                                  "n_baseline_samples": 100}),
        "GradCAM": GradCAM,
        "Gradient": Gradient
        })
