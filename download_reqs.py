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
    if args.all or "CIFAR-10" in args.datasets:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.4913, 0.4821, 0.4464), std=(0.247,0.2434, 0.2615))
        ])
        ds = datasets.CIFAR10(os.path.join("data", "CIFAR10"), train=False, transform=transform, download=True)
    if args.all or "CIFAR-100" in args.datasets:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5072, 0.4867, 0.441), std=[0.2673, 0.2565, 0.2762])
        ])
        ds = datasets.CIFAR100(os.path.join("data", "CIFAR100"), train=False, transform=transform, download=True)
    if args.all or "SVHN" in args.datasets:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4377, 0.4438, 0.4728), (0.1980, 0.2010, 0.1970)),
        ])
        ds = datasets.SVHN(os.path.join("data", "SVHN"), split="test", transform=transform, download=True)
    if args.all or "ImageNet" in args.datasets:
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        #TODO download and extract tgz file from github
    if args.all or "Places-365" in args.datasets:
        transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        ds = datasets.Places365(os.path.join("data", 'Places365'), split="val", transform=transform, small=True, download=True)
    if args.all or "Caltech-256" in args.datasets:
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.552, 0.5332, 0.5047], [0.3154, 0.312, 0.3256])
        ])
        #TODO check if downloading from pytorch gives the same result