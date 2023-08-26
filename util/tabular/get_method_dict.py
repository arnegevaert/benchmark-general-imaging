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
    method_dict = {
        "Gradient": Gradient(model),
        "InputXGradient": InputXGradient(model),
        "DeepLift": DeepLift(model),
        "IntegratedGradients": IntegratedGradients(
            model, internal_batch_size=32, num_steps=25
        ),
        "SmoothGrad": SmoothGrad(
            model, num_samples=50, stdev=0.15
        ),
        "VarGrad": VarGrad(model, num_samples=50, stdev=0.15),
        "DeepShap": DeepShap(
            model, reference_dataset=reference_dataset, num_baseline_samples=100
        ),
        "ExpectedGradients": ExpectedGradients(
            model, reference_dataset=reference_dataset, num_samples=100
        ),
        "Random": Random(model),
        "KernelSHAP": KernelShap(model, num_samples=300, super_pixels=False),
        "LIME": TabularLime(model, num_samples=300)
    }
    return method_dict