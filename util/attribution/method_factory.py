from typing import List
from util.attribution import (
    Deconvolution,
    DeepShap,
    DeepLift,
    ExpectedGradients,
    GradCAM,
    Gradient,
    InputXGradient,
    GuidedBackprop,
    Deconvolution,
    GuidedGradCAM,
    IntegratedGradients,
    SmoothGrad,
    VarGrad,
    KernelShap,
    ImageLime,
    Random,
    Rise,
    ExtremalPerturbation,
    Igos,
    Igos_pp,
    XRAI
)
from attribench import MethodFactory
from torch.utils.data import Dataset


def get_method_factory(
    batch_size: int,
    reference_dataset: Dataset,
    methods: List[str] | None = None,
) -> MethodFactory:
    config_dict = {
        "Gradient": Gradient,
        "InputXGradient": InputXGradient,
        "GuidedBackprop": GuidedBackprop,
        "Deconvolution": Deconvolution,
        "GradCAM": GradCAM,
        "GuidedGradCAM": GuidedGradCAM,
        "DeepLift": DeepLift,
        "IntegratedGradients": (
            IntegratedGradients,
            {"internal_batch_size": batch_size, "num_steps": 25},
        ),
        "SmoothGrad": (SmoothGrad, {"num_samples": 50, "stdev": 0.15}),
        "VarGrad": (VarGrad, {"num_samples": 50, "stdev": 0.15}),
        "DeepShap": (
            DeepShap,
            {
                "reference_dataset": reference_dataset,
                "num_baseline_samples": 100,
            },
        ),
        "KernelSHAP": (
            KernelShap,
            {"num_segments": 50, "num_samples": 300, "super_pixels": True},
        ),
        "ExpectedGradients": (
            ExpectedGradients,
            {"reference_dataset": reference_dataset, "num_samples": 100},
        ),
        "LIME": (ImageLime, {"num_segments": 50, "num_samples": 300}),
        "Random": Random,
        #"Rise":Rise,
        "Extremal_perturbation":ExtremalPerturbation,
        "IGOS": Igos,
        "IGOS_pp": Igos_pp,
        "XRAI":(XRAI,{'batch_size':batch_size}),
    }

    if methods is not None:
        filtered_config_dict = {}
        for method in methods:
            filtered_config_dict[method] = config_dict[method]
        return MethodFactory(filtered_config_dict)

    return MethodFactory(config_dict)
