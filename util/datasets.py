from torchvision import datasets, transforms
from os import path
import os
from PIL import Image
from torch.utils.data import Dataset
import pandas as pd


class VOCDataset(Dataset):
    classes = [
        "aeroplane",
        "bicycle",
        "bird",
        "boat",
        "bottle",
        "bus",
        "car",
        "cat",
        "chair",
        "cow",
        "diningtable",
        "dog",
        "horse",
        "motorbike",
        "person",
        "pottedplant",
        "sheep",
        "sofa",
        "train",
        "tvmonitor",
    ]

    def __init__(self, root, train, transform):
        super().__init__()
        self.root = root
        self.split = "train" if train else "val"
        self.transform = transform
        self.image_dir = path.join(self.root, "JPEGImages")
        splits_dir = path.join(self.root, "ImageSets/Main")

        split_f = path.join(splits_dir, self.split + ".txt")
        with open(split_f, "r") as f:
            self.file_names = [n.strip() for n in f.readlines()]

        annotations = {}
        for cls in VOCDataset.classes:
            with open(
                path.join(splits_dir, cls + "_" + self.split + ".txt")
            ) as f:
                annotations[cls] = {
                    line.split()[0]: line.split()[1] == "1"
                    for line in f.readlines()
                }
        self.annotations = pd.DataFrame.from_dict(annotations)
        assert not (
            self.annotations.isna().any().any()
            or self.annotations.isnull().any().any()
        )

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, item):
        name = self.file_names[item]
        f_loc = path.join(self.image_dir, name + ".jpg")
        label = self.annotations.loc[name].to_numpy(dtype="float32")

        img = Image.open(f_loc).convert("RGB")
        img = self.transform(img)

        return img, label


class Imagenette2Imagenet(datasets.ImageFolder):
    # Some changes for ImageFolder are needed to map the imagenette classes
    # back to the pytorch model output indices
    def __init__(self, root: str, transform=None):
        super().__init__(root, transform=transform)
        parent_dir = path.dirname(root)
        with open(path.join(parent_dir, "imagenet_classes.txt")) as f:
            all_classes = f.readlines()
        present_classes = os.listdir(root)
        self.classes = [x.strip() for x in all_classes]
        self.class_to_idx = {
            cls: idx
            for idx, cls in enumerate(self.classes)
            if cls in present_classes
        }
        samples = datasets.folder.make_dataset(
            self.root, self.class_to_idx, self.extensions
        )
        self.samples = samples
        self.imgs = samples
        self.targets = [s[1] for s in samples]


ALL_DATASETS = [
    "MNIST",
    "FashionMNIST",
    "CIFAR10",
    "CIFAR100",
    "SVHN",
    "ImageNet",
    "Places365",
    "Caltech256",
]
SAMPLE_SHAPES = {
    "MNIST": (1, 28, 28),
    "FashionMNIST": (1, 28, 28),
    "CIFAR10": (3, 32, 32),
    "CIFAR100": (3, 32, 32),
    "SVHN": (3, 32, 32),
    "ImageNet": (3, 224, 224),
    "Places365": (3, 224, 224),
    "Caltech256": (3, 224, 224),
}


def get_dataset(name, data_dir, train=False):
    assert name in ALL_DATASETS, f"Invalid dataset: {name}."
    if name == "MNIST":
        transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        )
        return datasets.MNIST(
            path.join(data_dir, "MNIST"),
            train=train,
            transform=transform,
            download=True,
        )
    elif name == "FashionMNIST":
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.2859635), (0.35296154)),
            ]
        )
        return datasets.FashionMNIST(
            path.join(data_dir, "FashionMNIST"),
            train=train,
            transform=transform,
            download=True,
        )
    elif name == "CIFAR10":
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.4913, 0.4821, 0.4464), std=(0.247, 0.2434, 0.2615)
                ),
            ]
        )
        return datasets.CIFAR10(
            path.join(data_dir, "CIFAR10"),
            train=train,
            transform=transform,
            download=True,
        )
    elif name == "CIFAR100":
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.5072, 0.4867, 0.441), std=[0.2673, 0.2565, 0.2762]
                ),
            ]
        )
        return datasets.CIFAR100(
            path.join(data_dir, "CIFAR100"),
            train=train,
            transform=transform,
            download=True,
        )
    elif name == "SVHN":
        split = "train" if train else "test"
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.4377, 0.4438, 0.4728), (0.1980, 0.2010, 0.1970)
                ),
            ]
        )
        return datasets.SVHN(
            path.join(data_dir, "SVHN"),
            split=split,
            transform=transform,
            download=True,
        )
    elif name == "ImageNet":
        transform = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
        dir = "train" if train else "val"
        return Imagenette2Imagenet(
            path.join(data_dir, "imagenette2", dir), transform=transform
        )
    elif name == "Places365":
        transform = transforms.Compose(
            [
                transforms.Resize((256, 256)),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
                ),
            ]
        )
        dir = "train" if train else "val"
        return datasets.Places365(
            path.join(data_dir, "Places365"),
            split=dir,
            transform=transform,
            small=True,
            download=False,
        )
    elif name == "Caltech256":
        transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    [0.552, 0.5332, 0.5047], [0.3154, 0.312, 0.3256]
                ),
            ]
        )
        dir = "Train" if train else "Test"
        return datasets.ImageFolder(
            path.join(data_dir, "Caltech256", dir), transform=transform
        )
    elif name == "VOC2012":
        transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    [0.4568, 0.4386, 0.4063], [0.2666, 0.2641, 0.2783]
                ),
            ]
        )
        return VOCDataset(
            path.join(data_dir, "VOC2012"), train=train, transform=transform
        )
