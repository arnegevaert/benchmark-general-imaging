import argparse
import time
import numpy as np
import torch
from experiments.general_imaging.lib.dataset_models import get_dataset_model
from attrbench.metrics import runtime
from experiments.lib import MethodLoader
from torch.utils.data import DataLoader
import os
import pandas as pd


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("method_config", type=str)
    parser.add_argument("-d", "--dataset", type=str, required=True)
    parser.add_argument("-m", "--model", type=str)
    parser.add_argument("-b", "--batch-size", type=int, required=True)
    parser.add_argument("-n", "--num-samples", type=int, required=True)
    parser.add_argument("-c", "--cuda", action="store_true")
    parser.add_argument("-o", "--output", type=str, required=True)
    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() and args.cuda else "cpu"

    ds, model, _ = get_dataset_model(args.dataset, model_name=args.model)
    model.eval()
    model.to(device)
    methods = MethodLoader(model=model, last_conv_layer=model.get_last_conv_layer(),
                           reference_dataset=ds).load_config(args.method_config)

    if not os.path.isdir(args.output):
        os.makedirs(args.output)

    num_done = 0
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=4)
    it = iter(dl)
    runtimes_batch = {m_name: [] for m_name in methods}
    runtimes_single = {m_name: [] for m_name in methods}
    while num_done < args.num_samples:
        batch, labels = next(it)
        batch = batch.to(device)
        labels = labels.to(device)

        time.sleep(5)

        for method_name, method in methods.items():
            print(method_name)
            # Evaluate runtime on batch
            runtimes_batch[method_name].append(
                runtime(batch, labels, method).detach().cpu().numpy()
            )
            runtimes_single[method_name].append(
                runtime(batch, labels, method, single_image=True).detach().cpu().numpy()
            )
        num_done += batch.size(0)
    runtimes_batch = pd.DataFrame(
        {method_name: np.concatenate(runtimes_batch[method_name]).squeeze() for method_name in runtimes_batch}
    )
    runtimes_single = pd.DataFrame(
        {method_name: np.concatenate(runtimes_single[method_name]).squeeze() for method_name in runtimes_single}
    )
    runtimes_batch.to_csv(os.path.join(args.output, "batch.csv"))
    runtimes_single.to_csv(os.path.join(args.output, "single.csv"))
