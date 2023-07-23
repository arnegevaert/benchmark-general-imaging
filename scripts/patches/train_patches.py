import argparse
from os import path
from util.models import ModelFactoryImpl
from util.datasets import get_dataset
from attribench.distributed import TrainAdversarialPatches


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--num-patches", type=int, required=True)
    parser.add_argument("--out-dir", type=str, required=True)
    args = parser.parse_args()

    model_factory = ModelFactoryImpl(args.dataset, args.data_dir, args.model)
    dataset = get_dataset(args.dataset, args.data_dir)
    train_adv_patches = TrainAdversarialPatches(
        model_factory, dataset, args.num_patches, args.batch_size, args.out_dir
    )
    train_adv_patches.run()
