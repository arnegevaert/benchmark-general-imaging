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
)


def get_method_dict(model, train_dataset):
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
            model, reference_dataset=train_dataset, num_baseline_samples=100
        ),
        "ExpectedGradients": ExpectedGradients(
            model, reference_dataset=train_dataset, num_samples=100
        ),
        "Random": Random(model),
        # TODO add KernelSHAP, LIME (as applicable)
    }
    return method_dict