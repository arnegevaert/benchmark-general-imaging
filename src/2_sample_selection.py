from typing import Optional
import os
from os import path
from models import BasicCNN, Resnet18, Resnet20, Resnet56, Resnet50
from imagenet_dataset import ImagenetDataset
from torchvision import datasets, transforms
from attrbench import Model
from attrbench.data import HDF5DatasetWriter
from attrbench.distributed import SampleSelection
import argparse


_DATASETS = ["MNIST", "CIFAR10", "CIFAR100", "SVHN", "ImageNet", "Places365",
             "Caltech256"]


def get_medium_dim_model(data_dir: str, model_name: str, num_classes: int,
                         ds_name: str):
        model_path = path.join(data_dir, f"models/{ds_name}/{model_name}.pt")
        if model_name == 'resnet20':
            model = Resnet20(num_classes, model_path)
        elif model_name == 'resnet56':
            model = Resnet56(num_classes, model_path)
        else:
            raise ValueError(f"Invalid model: {model_name}."
                             "Use resnet20 or resnet56.")
        return model


def get_high_dim_model(data_dir: str, model_name: str, num_classes: int,
                       ds_name: str):
        model_path = path.join(data_dir, f"models/{ds_name}/{model_name}.pt")
        if model_name == 'resnet18':
            model = Resnet18(num_classes, model_path)
        elif model_name == 'resnet50':
            model = Resnet50(num_classes, model_path)
        else:
            raise ValueError(f"Invalid model: {model_name}."
                             "Use resnet18 or resnet50.")
        return model


def get_mnist(data_dir: str):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    ds = datasets.MNIST(path.join(data_dir, "MNIST"), train=False,
                        transform=transform, download=True)
    model = BasicCNN(10, path.join(data_dir, "models/MNIST/cnn.pt"))
    return ds, model


def get_fashion_mnist(data_dir: str):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.2859635), (0.35296154)),
    ])
    ds = datasets.FashionMNIST(path.join(data_dir, "FashionMNIST"), 
                               train=False, transform=transform,
                               download=True)
    model = BasicCNN(10, path.join(data_dir, "models/FashionMNIST/cnn.pt"))
    return ds, model


def get_cifar10(data_dir: str, model_name: str):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.4913, 0.4821, 0.4464),
                             std=(0.247,0.2434, 0.2615))
    ])
    ds = datasets.CIFAR10(path.join(data_dir, "CIFAR10"), train=False,
                          transform=transform, download=True)
    model = get_medium_dim_model(data_dir, model_name, 10, "CIFAR10")
    return ds, model


def get_cifar100(data_dir: str, model_name: str):
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5072, 0.4867, 0.441),
                                 std=[0.2673, 0.2565, 0.2762])
        ])
        ds = datasets.CIFAR100(path.join(data_dir, "CIFAR100"), train=False,
                               transform=transform, download=True)
        model = get_medium_dim_model(data_dir, model_name, 100, "CIFAR100")
        return ds, model


def get_svhn(data_dir: str, model_name: str):
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4377, 0.4438, 0.4728),
                                 (0.1980, 0.2010, 0.1970)),
        ])
        ds = datasets.SVHN(path.join(data_dir, "SVHN"), split="test",
                           transform=transform, download=True)
        model = get_medium_dim_model(data_dir, model_name, 10, "SVHN")
        return ds, model


# TODO upload ImageNet model weights to zenodo so we can use the same
# function to get the model here (instead of setting pretrained=True)
def get_imagenet(data_dir: str, model_name: str):
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        ds = ImagenetDataset(path.join(data_dir, "imagenette2", "val"), transform=transform)
        if model_name.lower() == 'resnet18':
            model = Resnet18(1000, pretrained=True)
        elif model_name.lower() == 'resnet50':
            model = Resnet50(1000, pretrained=True)
        else:
            raise ValueError(f"Invalid model for this dataset: {model_name}")
        return ds, model


def get_places365(data_dir: str, model_name: str):
        transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        ds = datasets.Places365(path.join(data_dir, 'Places365', "val"),
                                split=dir, transform=transform, small=True,
                                download=False)
        model = get_high_dim_model(data_dir, model_name, 365, "Places365")
        return ds, model


def get_caltech256(data_dir: str, model_name: str):
        transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.552, 0.5332, 0.5047], [0.3154, 0.312, 0.3256])
        ])
        ds = datasets.ImageFolder(path.join(data_dir,'Caltech256', "Test"),
                                  transform=transform)
        model = get_high_dim_model(data_dir, model_name, 267, "Caltech256")
        return ds, model



def get_dataset_model(data_dir: str, dataset_name: str,
                      model_name: Optional[str] = None):
    if dataset_name == "MNIST":
        dataset, model = get_mnist(data_dir)
        sample_shape = (1, 28, 28)
    elif dataset_name == "FashionMNIST":
        dataset, model = get_fashion_mnist(data_dir)
        sample_shape = (1, 28, 28)
    else:
        if model_name is None:
            raise ValueError(f"model_name must be specified" 
                             f" for dataset {dataset_name}")
        if dataset_name == "CIFAR10":
            dataset, model = get_cifar10(data_dir, model_name)
            sample_shape = (3, 32, 32)
        elif dataset_name == "CIFAR100":
            dataset, model = get_cifar100(data_dir, model_name)
            sample_shape = (3, 32, 32)
        elif dataset_name == "SVHN":
            dataset, model = get_svhn(data_dir, model_name)
            sample_shape = (3, 32, 32)
        elif dataset_name == "ImageNet":
            dataset, model = get_imagenet(data_dir, model_name)
            sample_shape = (3, 224, 224)
        elif dataset_name=="Places365":
            dataset, model = get_places365(data_dir, model_name)
            sample_shape = (3, 224, 224)
        elif dataset_name=="Caltech256":
            dataset, model = get_caltech256(data_dir, model_name)
            sample_shape = (3, 224, 224)
        else:
            raise ValueError(f"Invalid dataset name {dataset_name}")
    return dataset, model, sample_shape


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--num-samples", type=int)
    parser.add_argument("-d" "--dataset", type=str, choices=_DATASETS)
    parser.add_argument("-b", "--batch-size", type=int)
    parser.add_argument("-o", "--output-file", type=str)
    parser.add_argument("-m", "--model", type=str, nargs="?")
    parser.add_argument("--data-dir", type=str)
    parser.add_argument("--allow-overwrite", action="store_true")
    args = parser.parse_args()

    dataset, model, sample_shape = get_dataset_model(
            args.data_dir, args.dataset, args.model)

    if path.exists(args.output_file):
        if args.allow_overwrite:
            os.remove(args.output_file)
        else:
            raise ValueError("Output file exists. Pass --allow-overwrite to"
                             " allow the script to overwrite this file.")

    writer = HDF5DatasetWriter(
            path=args.output_file,
            num_samples=args.num_samples, sample_shape=sample_shape)
    sample_selection = SampleSelection(Model(model), dataset , writer, num_samples=args.num_samples, batch_size=args.batch_size)
    sample_selection.run()
