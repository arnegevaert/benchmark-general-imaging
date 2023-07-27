# Evaluating Feature Attribution Methods in the Image Domain
This repository contains the code to reproduce the results presented in [Evaluating Feature Attribution Methods in the Image Domain](www.arxiv.org). Implementations of the metrics are available in the [`attribench` package](https://github.com/zoeparman/benchmark), which this repository depends on.

## Installation
Installing the required dependencies to reproduce and/or visualize the results can be done using a [virtual environment](https://docs.python.org/3/tutorial/venv.html):
```bash
$ python3 -m venv venv
$ source venv/bin/activate
$ pip install -r requirements.txt
```

## Reproducing the results
### 1. Download the datasets
To download all the datasets used in the original publication:
```bash
(venv) $ python download_reqs.py -d all
```
To download specific datasets, pass the desired datasets as arguments to the script. For example, if only MNIST, FashionMNIST and Places-365 are required:
```
(venv) $ python download_reqs.py -d MNIST FashionMNIST Places-365
```
**Note:** Instead of the full ImageNet dataset, this script will download [ImageNette](https://github.com/fastai/imagenette), a subset of ImageNet.

### 2. Download trained models
To download trained model parameters for the datasets and models used in the original publication:
```bash
(venv) $ python download_reqs.py -m
```

### 3. Train adversarial patches
**This step is optional:** The following command can be used to download trained adversarial patches for the datasets used in the original publication:
```bash
(venv) $ python download_reqs.py -p
```

If you prefer training the adversarial patches yourself, the `train_patches.py` script can be used. An example for the ImageNet dataset, using ResNet18 architecture, a batch size of 64, and CUDA:
```bash
(venv) $ python train_patches.py -d ImageNet -m resnet18 -b 64 -c
```
For more information, run `python train_patches.py -h`.

### 4. Run benchmark
**This step is optional:** The following command can be used to download benchmark results from the original publication:
```bash
(venv) $ python download_reqs.py -r
```

To run the general benchmark on a given dataset, use the `run_benchmark.py` script. This script requires 2 configuration files: one file specifying the metrics that need to be run (this configuration is passed to the `attribench` dependency package), and another specifying the attribution methods that need to be tested (this configuration is processed directly). The following commands can be used to run the full benchmark on all datasets from the original publication, for 256 samples using a batch size of 64:
```bash
(venv) $ python run_benchmark.py -d ImageNet -m resnet18 -b 64 -n 256 -ci -o imagenet.h5 config/suite.yaml config/methods.yaml
(venv) $ python run_benchmark.py -d Caltech256 -m resnet18 -b 64 -n 256 -ci -o caltech.h5 config/suite.yaml config/methods.yaml
(venv) $ python run_benchmark.py -d Places365 -m resnet18 -b 64 -n 256 -ci -o places.h5 config/suite.yaml config/methods.yaml
(venv) $ python run_benchmark.py -d MNIST -m CNN -b 64 -n 256 -ci -o mnist.h5 config/suite_no_ic.yaml config/methods.yaml
(venv) $ python run_benchmark.py -d FashionMNIST -m CNN -b 64 -n 256 -ci -o fashionmnist.h5 config/suite_no_ic.yaml config/methods.yaml
(venv) $ python run_benchmark.py -d SVHN -m resnet20 -b 64 -n 256 -ci -o svhn.h5 config/suite_no_ic.yaml config/methods.yaml
(venv) $ python run_benchmark.py -d CIFAR10 -m resnet20 -b 64 -n 256 -ci -o cifar10.h5 config/suite_no_ic.yaml config/methods.yaml
(venv) $ python run_benchmark.py -d CIFAR100 -m resnet20 -b 64 -n 256 -ci -o cifar100.h5 config/suite_no_ic.yaml config/methods.yaml
```

For more info on the arguments for the `run_benchmark.py` script, run `python run_benchmark.py -h`.

We provide 2 configuration files for the benchmark: `config/suite.yaml` and `config/suite_no_ic.yaml`. These files are exactly the same, except that `config/suite_no_ic.yaml` will skip the Impact Coverage metric. Use this script on the low-dimensional datasets, for which adversarial patches are not available.

`config/methods.yaml` contains the configuration for attribution methods. This file can be modified to remove/add attribution methods, or change their hyperparameters.

To run the signal-to-noise ratio experiments for Sensitivity-n and SegSensitivity-n (see Section 7.5 of the paper), use the `sens_n_variance.py` script. For example, for the MNIST dataset (other datasets are analogous):
```bash
(venv) $ python sens_n_variance.py -d MNIST -m CNN -b 64 -n 256 -i 100 -o out -c
```

### 5. Analyse results
To generate all plots from the paper, three scripts are used. These examples assume that the `.h5` files with the results are stored in the `out/` directory.
1. To produce the general plots: `python plot/general_plots.py out/ plot/out`
2. To produce the plots that compare the SNR and variance of Sensitivity-n and SegSensitivity-n: `python plot/sens_n_variance_plots.py out/ plot/out/sens_n.png`
3. To produce the specific plots for the case study on ImageNet (appendix): `python plot/case_study_plots.py out/ plot/out/case_study`