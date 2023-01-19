import os.path as path
from torchvision import datasets
from torchvision.datasets.folder import make_dataset


class ImagenetDataset(datasets.ImageFolder):
    # Some changes for ImageFolder are needed to map the imagenette classes back to the pytorch model output indices
    def __init__(self, root: str, transform=None):
        super().__init__(root, transform=transform)
        parent_dir = path.dirname(root)
        with open(path.join(parent_dir, "imagenet_classes.txt")) as f:
            classes = f.readlines()
        self.classes = [x.strip() for x in classes]
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        samples = make_dataset(self.root, self.class_to_idx, self.extensions)
        self.samples = samples
        self.imgs = samples
        self.targets = [s[1] for s in samples]
