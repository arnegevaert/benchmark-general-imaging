from torchvision import datasets, transforms
import os
from os import path
import torch
import torch.nn as nn
from torch.nn import functional as F
from lib.models import Resnet20, Resnet18, Resnet56, Resnet50
from lib.datasets import ImagenetDataset


_DATA_LOC = os.environ["BM_DATA_LOC"] if "BM_DATA_LOC" in os.environ else path.join(path.dirname(__file__), "../data")


def get_dataset_model(name, model_name=None, train=False):
    if name == "MNIST":
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        ds = datasets.MNIST(path.join(_DATA_LOC, "MNIST"), train=train, transform=transform, download=True)
        model = BasicCNN(10, path.join(_DATA_LOC, "models/MNIST/cnn.pt"))
        patch_folder = None
    elif name == "FashionMNIST":
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.2859635), (0.35296154)),
        ])
        ds = datasets.FashionMNIST(path.join(_DATA_LOC, "FashionMNIST"), train=train, transform=transform, download=True)
        model = BasicCNN(10, path.join(_DATA_LOC, "models/FashionMNIST/cnn.pt"))
        patch_folder = None
    elif name == "CIFAR10":
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.4913, 0.4821, 0.4464), std=(0.247,0.2434, 0.2615))
        ])
        ds = datasets.CIFAR10(path.join(_DATA_LOC, "CIFAR10"), train=train, transform=transform, download=True)
        if model_name.lower() == 'resnet20':
            model = Resnet20(10,path.join(_DATA_LOC, "models/CIFAR10/resnet20.pt"))
        elif model_name.lower() == 'resnet56':
            model = Resnet56(10, path.join(_DATA_LOC, "models/CIFAR10/resnet56.pt"))
        else:
            raise ValueError(f"Invalid model for this dataset: {model_name}")
        patch_folder = path.join(_DATA_LOC, "patches/CIFAR10", model_name.lower())
    elif name == "CIFAR100":
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5072, 0.4867, 0.441), std=[0.2673, 0.2565, 0.2762])
        ])
        ds = datasets.CIFAR100(path.join(_DATA_LOC, "CIFAR100"), train=train, transform=transform, download=True)
        if model_name.lower() == 'resnet20':
            model = Resnet20(100, path.join(_DATA_LOC, "models/CIFAR100/resnet20.pt"))
        elif model_name.lower() == 'resnet56':
            model = Resnet56(100, path.join(_DATA_LOC, "models/CIFAR100/resnet56.pt"))
        else:
            raise ValueError(f"Invalid model for this dataset: {model_name}")
        patch_folder = path.join(_DATA_LOC, "patches/CIFAR100", model_name)

    elif name == "SVHN":
        split = "train" if train else "test"
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4377, 0.4438, 0.4728), (0.1980, 0.2010, 0.1970)),
        ])
        ds = datasets.SVHN(path.join(_DATA_LOC, "SVHN"), split=split, transform=transform, download=True)
        if model_name.lower() == 'resnet20':
            model = Resnet20(10, path.join(_DATA_LOC, "models/SVHN/resnet20.pt"))
        elif model_name.lower() == 'resnet56':
            model = Resnet56(10, path.join(_DATA_LOC, "models/SVHN/resnet56.pt"))
        else:
            raise ValueError(f"Invalid model for this dataset: {model_name}")
        patch_folder = path.join(_DATA_LOC, "patches/SVHN", model_name.lower())

    elif name == "ImageNet":
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        dir = "train" if train else "val"
        ds = ImagenetDataset(path.join(_DATA_LOC, "ImageNet", "imagenette2", dir), transform=transform)
        if model_name.lower() == 'resnet18':
            model = Resnet18(1000, pretrained=True)
        elif model_name.lower() == 'resnet50':
            model = Resnet50(1000, pretrained=True)
        else:
            raise ValueError(f"Invalid model for this dataset: {model_name}")
        patch_folder = path.join(_DATA_LOC, "patches/ImageNet", model_name.lower())

    elif name=="Places365":
        transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        dir = "train" if train else "val"
        ds = datasets.Places365(path.join(_DATA_LOC, 'Places365'), split=dir, transform=transform, small=True,
                                download=False)
        if model_name.lower() == 'resnet18':
            model = Resnet18(365,params_loc = path.join(_DATA_LOC, "models/Places365/resnet18.pt"),pretrained=False)
        elif model_name.lower() == 'resnet50':
            model = Resnet50(365,params_loc = path.join(_DATA_LOC, "models/Places365/resnet50.pt"),pretrained=False)
        else:
            raise ValueError(f"Invalid model for this dataset: {model_name}")
        patch_folder = path.join(_DATA_LOC, "patches/Places365", model_name.lower())
    elif name=="Caltech256":
        transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.552, 0.5332, 0.5047], [0.3154, 0.312, 0.3256])
        ])
        dir = "Train" if train else "Test"
        ds = datasets.ImageFolder(path.join(_DATA_LOC,'Caltech256',dir), transform=transform)
        if model_name.lower() == 'resnet18':
            model = Resnet18(267, params_loc=path.join(_DATA_LOC, "models/Caltech256/resnet18.pt"))
        elif model_name.lower() == 'resnet50':
            model = Resnet50(267, params_loc=path.join(_DATA_LOC, "models/Caltech256/resnet50.pt"))
        else:
            raise ValueError(f"Invalid model for this dataset: {model_name}")
        patch_folder = path.join(_DATA_LOC, "patches/Caltech256", model_name.lower())
    else:
        raise ValueError(f"Invalid dataset: {name}")

    return ds, model, patch_folder


class BasicCNN(nn.Module):
    """
    Basic convolutional network for MNIST
    """
    def __init__(self, num_classes, params_loc=None):
        super(BasicCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        if params_loc:
            # map_location allows taking a model trained on GPU and loading it on CPU
            # without it, a model trained on GPU will be loaded in GPU even if DEVICE is CPU
            self.load_state_dict(torch.load(params_loc, map_location=lambda storage, loc: storage))

    def forward(self, x):
        if type(x) != torch.Tensor:
            x = torch.tensor(x)
        if x.dtype != torch.float32:
            x = x.float()

        relu = nn.ReLU()
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.relu2(x)
        x = self.dropout2(x)
        return self.fc2(x)

    def get_last_conv_layer(self):
        return self.conv2
