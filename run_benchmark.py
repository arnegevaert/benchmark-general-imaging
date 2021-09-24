import argparse
import torch
from torch.utils.data import DataLoader, TensorDataset
from lib.dataset_models import get_dataset_model
from lib import MethodLoader
from attrbench.suite import Suite, MetricLoader, SuiteResult
import logging

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("suite_config", type=str)
    parser.add_argument("method_config", type=str)
    parser.add_argument("-d", "--dataset", type=str, required=True)
    parser.add_argument("-m", "--model", type=str)
    parser.add_argument("-b", "--batch-size", type=int, required=True)
    parser.add_argument("-n", "--num-samples", type=int, required=True)
    parser.add_argument("-c", "--cuda", action="store_true")
    parser.add_argument("-i", "--save-images", action="store_true")
    parser.add_argument("-a", "--save-attrs", action="store_true")
    parser.add_argument("-o", "--output", type=str, required=True)
    parser.add_argument("--log-dir", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--multi_label", action="store_true")
    parser.add_argument("--explain_label", type=int, default=None)
    parser.add_argument("--num_baseline", type=int, default=25)
    parser.add_argument("--source_hdf", type=str, default=None)
    # Parse arguments
    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() and args.cuda else "cpu"

    logging.basicConfig(
        format='[%(asctime)s %(levelname)s] %(message)s',
        level=logging.INFO,
        datefmt="%Y-%m-%d %H:%M:%S")

    logging.info("Saving images" if args.save_images else "Not saving images")
    logging.info("Saving attributions" if args.save_images else "Not saving attributions")

    # Get dataset, model, methods
    ds, model, patch_folder = get_dataset_model(args.dataset, model_name=args.model)
    methods = MethodLoader(model=model, last_conv_layer=model.get_last_conv_layer(),
                           reference_dataset=ds).load_config(args.method_config)

    # Get metrics
    metrics = MetricLoader(args.suite_config, model, methods, args.log_dir, patch_folder=patch_folder).load()

    # Get Dataloader
    if args.source_hdf is not None:
        print(f"Using images from {args.source_hdf}")
        res = SuiteResult.load_hdf(args.source_hdf)
        ds = TensorDataset(torch.tensor(res.images))

    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=4)

    # Run BM suite and save result to disk
    bm_suite = Suite(model, methods, metrics, device, log_dir=args.log_dir, multi_label=args.multi_label,
                     explain_label=args.explain_label, num_baseline_samples=args.num_baseline)
    bm_suite.run(dl, args.num_samples, args.seed, args.save_images, args.save_attrs, args.output)
