from attribench import MethodFactory
from util.attribution import (
    Gradient,
    InputXGradient,
    DeepLift,
    IntegratedGradients,
    SmoothGrad,
    VarGrad,
    DeepShap,
    ExpectedGradients,
    Random,
    KernelShap,
    TabularLime
)


def get_method_dict(model, reference_dataset):
    method_factory = get_method_factory(reference_dataset)
    return method_factory(model)


def get_method_factory(reference_dataset):
    config_dict = {
        "Gradient": Gradient,
        "InputXGradient": InputXGradient,
        "DeepLift": DeepLift,
        "IntegratedGradients": (
            IntegratedGradients,
            {"internal_batch_size": 32, "num_steps": 25},
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
        "ExpectedGradients": (
            ExpectedGradients,
            {"reference_dataset": reference_dataset, "num_samples": 100},
        ),
        "LIME": (TabularLime, {"num_samples": 300}),
        "KernelSHAP": (KernelShap, {"num_samples": 300, "super_pixels": False}),
        "Random": Random,
    }
    return MethodFactory(config_dict)