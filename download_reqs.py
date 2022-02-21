import zipfile
import requests
from io import BytesIO
import os
import argparse
from torchvision import datasets, transforms

_ALL_DATASETS = ["MNIST", "FashionMNIST", "CIFAR-10", "CIFAR-100", "SVHN", "ImageNet", "Places-365", "Caltech-256"]


def download_zip(name, url):
    print(f"Downloading {name}...")
    req = requests.get(url)
    print("Download completed")
    
    file = zipfile.ZipFile(BytesIO(req.content))
    file.extractall(f"./data/{name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-a", "--all", action="store_true")
    parser.add_argument("-p", "--patches", action="store_true")
    parser.add_argument("-m", "--models", action="store_true")
    parser.add_argument("-r", "--results", action="store_true")
    parser.add_argument("-d", "--datasets", nargs="*", choices=_ALL_DATASETS)
    args = parser.parse_args()

    if len(args.datasets) == 0:
        args.datasets = _ALL_DATASETS

    if not os.path.exists("./data"):
        os.makedirs("./data")

    # Download files from Zenodo
    if args.all or args.models:
        download_zip("models", "https://zenodo.org/record/6205531/files/models.zip?download=1")
    if args.all or args.patches:
        download_zip("patches", "https://zenodo.org/record/6205531/files/patches.zip?download=1")
    if args.all or args.results:
        download_zip("results", "https://zenodo.org/record/6205531/files/results.zip?download=1")

    # Download datasets using Pytorch
    if args.all or "MNIST" in args.datasets:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        datasets.MNIST(os.path.join("data", "MNIST"), train=False, transform=transform, download=True)
    if args.all or "FashionMNIST" in args.datasets:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.2859635), (0.35296154)),
        ])
        datasets.FashionMNIST(os.path.join("data", "FashionMNIST"), train=False, transform=transform, download=True)