import argparse
import time
from attribench.data import HDF5Dataset
from util.datasets import get_dataset
from util import attribution
from util.models import get_model
from torch.utils.data import DataLoader
import logging


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str)
    parser.add_argument("--samples-file", type=str)
    parser.add_argument("--batch-size", type=int)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    ds_name = "ImageNet"
    model_name = "resnet18"

    dataset = HDF5Dataset(args.samples_file)
    reference_dataset = get_dataset(ds_name, args.data_dir)
    model = get_model(ds_name, args.data_dir, model_name)
    model.to("cuda")

    methods = {
        "Gradient": attribution.Gradient(model),
        "CGI": attribution.CGI(model),
        "GradCAM++": attribution.GradCAMPP(model),
        "ScoreCAM": attribution.ScoreCAM(model),
        #"Integrated Gradients": attribution.IntegratedGradients(
        #    model, args.batch_size, num_steps=25
        #),
        #"Expected Gradients": attribution.ExpectedGradients(model, reference_dataset, num_samples=100),
        #"IGOS": attribution.Igos(model),
        #"IGOS++": attribution.Igos_pp(model),
        #"Extremal Perturbation": attribution.ExtremalPerturbation(model),
        "XRAI": attribution.XRAI(model, args.batch_size)
    }

    dl = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    batch_x, batch_y = next(iter(dl))
    batch_x = batch_x.to("cuda")
    batch_y = batch_y.to("cuda")

    for method_name, method in methods.items():
        start_t = time.time()
        print(f"Computing attributions for {method_name}...")
        attrs = method(batch_x, batch_y)
        end_t = time.time()
        total_time = end_t - start_t
        time_per_sample = total_time / batch_x.shape[0]
        print(
            f"Done in {total_time:.2f} seconds ({time_per_sample:.2f} s/sample)"
        )
        if attrs.shape != batch_x.shape:
            print(f"Wrong shape: expected {batch_x.shape}, got {attrs.shape}")
