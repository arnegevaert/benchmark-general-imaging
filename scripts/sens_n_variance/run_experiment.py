import torch
import argparse
import pandas as pd
from tqdm import tqdm
import os
from attribench.data import HDF5Dataset, AttributionsDataset
from attribench.functional.metrics import sensitivity_n
from attribench.masking.image import ConstantImageMasker
from util.datasets import ALL_DATASETS
from util.models import ModelFactoryImpl
from torch import multiprocessing as mp
from functools import partial


def _run_proc(
    model_factory,
    attrs_dataset,
    batch_size,
    maskers,
    activation_fns,
    min_subset_size,
    max_subset_size,
    num_steps,
    num_subsets,
    segmented,
    num_iterations,
    device,
    queue,
):
    model = model_factory()
    model.to(device)
    model.eval()

    result = []
    prog = tqdm(total=num_iterations, desc=f"Device {device}", position=device)
    for _ in range(num_iterations):
        sens_n_result = sensitivity_n(
            model,
            attrs_dataset,
            batch_size,
            maskers,
            activation_fns,
            min_subset_size,
            max_subset_size,
            num_steps,
            num_subsets,
            segmented,
            device,
        )
        df = sens_n_result.get_df(masker="constant", activation_fn="linear")[0]
        result.append(df)
        prog.update()
    prog.close()
    queue.put(pd.concat(result))


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

    # Build kwargs dict for each process
    devices = list(range(torch.cuda.device_count()))
    num_iterations = args.iterations // len(devices)
    nums_iterations = [num_iterations] * len(devices)
    # Increment num_iterations for each process until the total number of
    # iterations is correct (if args.iterations is not divisible by the number
    # of devices)
    for i in range(args.iterations % len(devices)):
        nums_iterations[i] += 1
    assert sum(nums_iterations) == args.iterations

    kwargs = [
        {
            "device": devices[i],
            "num_iterations": nums_iterations[i],
        }
        for i in range(len(devices))
    ]

    for segmented in [True, False]:
        ctx = mp.get_context("spawn")
        result_queue = ctx.Queue()
        processes = []
        for i in range(len(devices)):
            p = ctx.Process(
                target=_run_proc,
                kwargs=dict(
                    model_factory=model_factory,
                    attrs_dataset=attrs_dataset,
                    batch_size=args.batch_size,
                    maskers={
                        "constant": ConstantImageMasker(masking_level="pixel")
                    },
                    activation_fns=["linear"],
                    min_subset_size=0.1,
                    max_subset_size=0.5,
                    num_steps=10,
                    num_subsets=100,
                    segmented=segmented,
                    num_iterations=nums_iterations[i],
                    device=devices[i],
                    queue=result_queue,
                ),
            )
            p.start()
            processes.append(p)
        # Wait until processes are done
        for p in processes:
            p.join()

        # Gather results
        result = []
        for _ in range(len(devices)):
            result.append(result_queue.get())
        assert len(result) == len(devices)

        # Results don't need to be sorted because we parallelize over iterations
        # and each iteration is independent of the others
        result_df = pd.concat(result)
        if segmented:
            result_df.to_csv(
                os.path.join(args.out_dir, "seg_sens_n.csv"),
                index_label="Sample",
            )
        else:
            result_df.to_csv(
                os.path.join(args.out_dir, "sens_n.csv"), index_label="Sample"
            )
