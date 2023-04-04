from PIL import Image
import os
from torch.utils.data import Dataset
import pandas as pd


class VOCDataset(Dataset):
    classes = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat',
               'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person',
               'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']

    def __init__(self, root, train, transform):
        super().__init__()
        self.root = root
        self.split = 'train' if train else 'val'
        self.transform = transform
        self.image_dir = os.path.join(self.root, 'JPEGImages')
        splits_dir = os.path.join(self.root, 'ImageSets/Main')

        split_f = os.path.join(splits_dir, self.split + '.txt')
        with open(split_f, 'r') as f:
            self.file_names = [n.strip() for n in f.readlines()]

        annotations = {}
        for cls in VOCDataset.classes:
            with open(os.path.join(splits_dir, cls + '_' + self.split + '.txt')) as f:
                annotations[cls] = {line.split()[0]: line.split()[1] == '1' for line in f.readlines()}
        self.annotations = pd.DataFrame.from_dict(annotations)
        assert (not (self.annotations.isna().any().any() or self.annotations.isnull().any().any()))

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, item):
        name = self.file_names[item]
        f_loc = os.path.join(self.image_dir, name + '.jpg')
        label = self.annotations.loc[name].to_numpy(dtype='float32')

        img = Image.open(f_loc).convert('RGB')
        img = self.transform(img)

        return img, label
