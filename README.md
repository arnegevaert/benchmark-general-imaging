# Evaluating Feature Attribution Methods in the Image Domain
This repository contains the code to reproduce the results presented in [Evaluating Feature Attribution Methods in the Image Domain](www.arxiv.org). Implementations of the metrics are available in the [`attrbench` package](https://github.com/zoeparman/benchmark), which this repository depends on.

## Reproducing the results
After downloading this repository and installing the dependencies in `requirements.txt`, the following steps can be used to reproduce the results from the paper:
1. **Downloading the datasets:** only the high-dimensional datasets need to be downloaded before running, the others are downloaded automatically.
   1. ImageNet: Any subset of ImageNet can be used. For the results in the paper, we used [ImageNette](https://github.com/fastai/imagenettehttps://github.com/fastai/imagenette), a subset of 10 classes from ImageNet. The model was trained on the full ImageNet dataset.
   2. Caltech256: http://www.vision.caltech.edu/Image_Datasets/Caltech256/
   3. Places365: http://places2.csail.mit.edu/download.html
2. **Computing the metric scores:** the `run_benchmark.py` script computes the metric scores for a single dataset and model. run `python run_benchmark.py --help` for more info. To compute metric scores for all the datasets:
```bash
python run_benchmark.py -d ImageNet -m resnet18 -b 64 -n 256 -ci -o imagenet.h5 config/suite.yaml config/methods.yaml
python run_benchmark.py -d Caltech256 -m resnet18 -b 64 -n 256 -ci -o caltech.h5 config/suite.yaml config/methods.yaml
python run_benchmark.py -d Places365 -m resnet18 -b 64 -n 256 -ci -o places.h5 config/suite.yaml config/methods.yaml
python run_benchmark.py -d MNIST -m CNN -b 64 -n 256 -ci -o mnist.h5 config/suite_no_ic.yaml config/methods.yaml
python run_benchmark.py -d FashionMNIST -m CNN -b 64 -n 256 -ci -o fashionmnist.h5 config/suite_no_ic.yaml config/methods.yaml
python run_benchmark.py -d SVHN -m resnet20 -b 64 -n 256 -ci -o svhn.h5 config/suite_no_ic.yaml config/methods.yaml
python run_benchmark.py -d CIFAR10 -m resnet20 -b 64 -n 256 -ci -o cifar10.h5 config/suite_no_ic.yaml config/methods.yaml
python run_benchmark.py -d CIFAR100 -m resnet20 -b 64 -n 256 -ci -o cifar100.h5 config/suite_no_ic.yaml config/methods.yaml
```

## Generating the plots
To generate all plots from the paper, three scripts are used. These examples assume that the `.h5` files with the results are stored in the `out/` directory.
1. To produce the general plots: `python plot/general_plots.py out/ plot/out`
2. To produce the plots that compare the SNR and variance of Sensitivity-n and SegSensitivity-n: `python plot/sens_n_variance_plots.py out/ plot/out/sens_n.png`
3. To produce the specific plots for the case study on ImageNet (appendix): `python plot/case_study_plots.py out/ plot/out/case_study`