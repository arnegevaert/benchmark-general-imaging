#!/bin/bash
source ~/anaconda3/etc/profile.d/conda.sh
conda activate benchmark

python run_benchmark.py -d ImageNet -m resnet18 -b 64 -n 256 -ci -o imagenet.h5 config/suite.yaml config/methods.yaml
python run_benchmark.py -d Caltech256 -m resnet18 -b 64 -n 256 -ci -o caltech.h5 config/suite.yaml config/methods.yaml
python run_benchmark.py -d Places365 -m resnet18 -b 64 -n 256 -ci -o places.h5 config/suite.yaml config/methods.yaml
python run_benchmark.py -d MNIST -m CNN -b 64 -n 256 -ci -o mnist.h5 config/suite_no_ic.yaml config/methods.yaml
python run_benchmark.py -d FashionMNIST -m CNN -b 64 -n 256 -ci -o fashionmnist.h5 config/suite_no_ic.yaml config/methods.yaml
python run_benchmark.py -d SVHN -m resnet20 -b 64 -n 256 -ci -o svhn.h5 config/suite_no_ic.yaml config/methods.yaml
python run_benchmark.py -d CIFAR10 -m resnet20 -b 64 -n 256 -ci -o cifar10.h5 config/suite_no_ic.yaml config/methods.yaml
python run_benchmark.py -d CIFAR100 -m resnet20 -b 64 -n 256 -ci -o cifar100.h5 config/suite_no_ic.yaml config/methods.yaml
