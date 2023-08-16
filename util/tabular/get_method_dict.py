from util import attribution


def get_method_dict(model, train_dataset):
    method_dict = {
        "Gradient": attribution.Gradient(model),
        "InputXGradient": attribution.InputXGradient(model),
        "DeepLift": attribution.DeepLift(model),
        "IntegratedGradients": attribution.IntegratedGradients(
            model, internal_batch_size=32, num_steps=25
        ),
        "SmoothGrad": attribution.SmoothGrad(
            model, num_samples=50, stdev=0.15
        ),
        "VarGrad": attribution.VarGrad(model, num_samples=50, stdev=0.15),
        "DeepShap": attribution.DeepShap(
            model, reference_dataset=train_dataset, num_baseline_samples=100
        ),
        "ExpectedGradients": attribution.ExpectedGradients(
            model, reference_dataset=train_dataset, num_samples=100
        ),
        "Random": attribution.Random(model),
        # TODO add RISE, extremal perturbation, XRAI, KernelSHAP, LIME (as applicable)
    }
    return method_dict