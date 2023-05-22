from os import path
from util.datasets import ALL_DATASETS
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.hub import load_state_dict_from_url


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
    'resnext50_32x4d': 'https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth',
    'resnext101_32x8d': 'https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth',
    'wide_resnet50_2': 'https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth',
    'wide_resnet101_2': 'https://download.pytorch.org/models/wide_resnet101_2-32ee1156.pth',
}


class BasicCNN(nn.Module):
    """
    Basic convolutional network for MNIST
    """

    def __init__(self, num_classes, params_loc=None):
        super(BasicCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        if params_loc:
            # map_location allows taking a model trained on GPU and loading it on CPU
            # without it, a model trained on GPU will be loaded in GPU even if DEVICE is CPU
            self.load_state_dict(torch.load(
                params_loc, map_location=lambda storage, loc: storage))

    def forward(self, x):
        if type(x) != torch.Tensor:
            x = torch.tensor(x)
        if x.dtype != torch.float32:
            x = x.float()

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


class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError(
                'BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError(
                "Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu2(out)

        return out


class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu1 = nn.ReLU(inplace=True)
        self.relu2 = nn.ReLU(inplace=True)
        self.relu3 = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu3(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000, small_model=False, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.small_model = small_model

        self.inplanes = 64 if not small_model else 16
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        k, s, p = (7, 2, 3) if not small_model else (3, 1, 1)
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=k, stride=s, padding=p,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        if not self.small_model:
            self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        if small_model:
            self.layer1 = self._make_layer(block, 16, layers[0])
            self.layer2 = self._make_layer(block, 32, layers[1], stride=2,
                                           dilate=replace_stride_with_dilation[0], option='A')
            self.layer3 = self._make_layer(block, 64, layers[2], stride=2,
                                           dilate=replace_stride_with_dilation[1], option='A')
            self.fc = nn.Linear(64 * block.expansion, num_classes)
        else:
            self.layer1 = self._make_layer(block, 64, layers[0])
            self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                           dilate=replace_stride_with_dilation[0])
            self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                           dilate=replace_stride_with_dilation[1])
            self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                           dilate=replace_stride_with_dilation[2])
            self.fc = nn.Linear(512 * block.expansion, num_classes)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False, option='B'):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            if option == 'A':
                downsample = LambdaLayer(lambda x:
                                         F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, planes // 4, planes // 4), "constant",
                                               0))
            else:
                downsample = nn.Sequential(
                    conv1x1(self.inplanes, planes * block.expansion, stride),
                    norm_layer(planes * block.expansion),
                )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def _forward_impl(self, x):
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        if not self.small_model:
            x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        if not self.small_model:
            x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

    def forward(self, x):
        return self._forward_impl(x)


class Resnet20(ResNet):
    def __init__(self, num_classes, params_loc=None):
        super().__init__(BasicBlock, [3, 3, 3],
                         num_classes=num_classes, small_model=True)
        if params_loc:
            state_dict = torch.load(
                params_loc, map_location=lambda storage, loc: storage)
            self.load_state_dict(state_dict)

    def get_last_conv_layer(self) -> nn.Module:
        return self.layer3[-1]


class Resnet32(ResNet):
    def __init__(self, num_classes, params_loc=None):
        super().__init__(BasicBlock, [5, 5, 5],
                         num_classes=num_classes, small_model=True)
        if params_loc:
            state_dict = torch.load(
                params_loc, map_location=lambda storage, loc: storage)
            self.load_state_dict(state_dict)

    def get_last_conv_layer(self) -> nn.Module:
        return self.layer3[-1]


class Resnet44(ResNet):
    def __init__(self, num_classes, params_loc=None):
        super().__init__(BasicBlock, [7, 7, 7],
                         num_classes=num_classes, small_model=True)
        if params_loc:
            state_dict = torch.load(
                params_loc, map_location=lambda storage, loc: storage)
            self.load_state_dict(state_dict)

    def get_last_conv_layer(self) -> nn.Module:
        return self.layer3[-1]


class Resnet56(ResNet):
    def __init__(self, num_classes, params_loc=None):
        super().__init__(BasicBlock, [9, 9, 9],
                         num_classes=num_classes, small_model=True)
        if params_loc:
            state_dict = torch.load(
                params_loc, map_location=lambda storage, loc: storage)
            self.load_state_dict(state_dict)

    def get_last_conv_layer(self) -> nn.Module:
        return self.layer3[-1]


class Resnet18_old(nn.Module):
    """
    Wrapper class around torchvision Resnet18 model,
    with 10 output classes and function to get the last convolutional layer
    """

    def __init__(self, num_classes=10, params_loc=None):
        super().__init__()
        self.model = ResNet(BasicBlock, [2, 2, 2, 2])
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

        if params_loc:
            self.model.load_state_dict(torch.load(
                params_loc, map_location=lambda storage, loc: storage))

    def forward(self, x):
        return self.model(x)

    def get_last_conv_layer(self):
        last_block = self.model.layer4[-1]  # Last BasicBlock of layer 3
        return last_block.conv2


class Resnet50(ResNet):
    def __init__(self, num_classes=1000, params_loc=None, pretrained=False):
        if pretrained:
            super().__init__(Bottleneck, [3, 4, 6, 3])
            state_dict = load_state_dict_from_url(model_urls['resnet50'])
            self.load_state_dict(state_dict)
            if num_classes != 1000:
                num_ftrs = self.fc.in_features
                self.fc = nn.Linear(num_ftrs, num_classes)
        else:
            super().__init__(Bottleneck, [3, 4, 6, 3], num_classes=num_classes)
        if params_loc:
            self.load_state_dict(torch.load(
                params_loc, map_location=lambda storage, loc: storage))

    def get_last_conv_layer(self):
        last_block = self.layer4[-1]  # Last BasicBlock of layer 3
        return last_block.conv2


class Resnet18(ResNet):
    def __init__(self, num_classes=1000, params_loc=None, pretrained=False):
        if pretrained:
            super().__init__(BasicBlock, [2, 2, 2, 2])
            state_dict = load_state_dict_from_url(model_urls['resnet18'])
            self.load_state_dict(state_dict)
            if num_classes != 1000:
                num_ftrs = self.fc.in_features
                self.fc = nn.Linear(num_ftrs, num_classes)
        else:
            super().__init__(BasicBlock, [2, 2, 2, 2], num_classes=num_classes)
        if params_loc:
            self.load_state_dict(torch.load(
                params_loc, map_location=lambda storage, loc: storage))

    def get_last_conv_layer(self):
        last_block = self.layer4[-1]  # Last BasicBlock of layer 3
        return last_block.conv2


def get_model(dataset_name, data_dir, model_name=None):
    assert dataset_name in ALL_DATASETS, f"Invalid dataset: {dataset_name}."
    if dataset_name in ["MNIST", "FashionMNIST"]:
        model_path = path.join(data_dir, f"models/{dataset_name}/cnn.pt")
        return BasicCNN(10, model_path)
    elif dataset_name in ["CIFAR10", "CIFAR100", "SVHN"]:
        assert model_name.lower() in ["resnet20", "resnet56"],\
            f"Invalid model for this dataset: {model_name}"
        model_path = path.join(
            data_dir, f"models/{dataset_name}/{model_name.lower()}.pt")
        return Resnet18(10, model_path)
    else:
        assert model_name.lower() in ["resnet18", "resnet50"],\
            f"Invalid model for this dataset: {model_name}"
        model_path = path.join(
            data_dir, f"models/{dataset_name}/{model_name.lower()}.pt")
        n_classes = {
            "ImageNet": 1000, "Places365": 365, "Caltech256": 267, "VOC2012": 20
        }
        pretrained = dataset_name == "ImageNet"
        if model_name.lower() == "resnet18":
            return Resnet18(n_classes[dataset_name], model_path, pretrained)
        elif model_name.lower() == "resnet50":
            return Resnet50(n_classes[dataset_name], model_path, pretrained)
