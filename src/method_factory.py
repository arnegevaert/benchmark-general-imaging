from attribution import Deconvolution, DeepShap, DeepLift,\
    ExpectedGradients, GradCAM, Gradient, InputXGradient, GuidedBackprop,\
    Deconvolution, GuidedGradCAM, IntegratedGradients, SmoothGrad,\
    VarGrad, KernelShap, ImageLime
from attrbench import MethodFactory

def get_method_factory(batch_size, dataset):
    return MethodFactory({
            "Gradient": Gradient,
            "InputXGradient": InputXGradient,
            "GuidedBackprop": GuidedBackprop,
            "Deconvolution": Deconvolution,
            "GradCAM": GradCAM,
            "GuidedGradCAM": GuidedGradCAM,
            "DeepLift": DeepLift,
            "IntegratedGradients": (IntegratedGradients,
                                    {"internal_batch_size": batch_size,
                                    "num_steps": 25}),
            "SmoothGrad": (SmoothGrad, {"num_samples": 50, "stdev": 0.15}),
            "VarGrad": (VarGrad, {"num_samples": 50, "stdev": 0.15}),
            "DeepShap": (DeepShap, {"reference_dataset": dataset,
                                    "num_baseline_samples": 100}),
            "KernelSHAP": (KernelShap, {"num_segments": 50,
                                        "num_samples": 300,
                                        "super_pixels": True}),
            "ExpectedGradients": (ExpectedGradients, {"reference_dataset": dataset,
                                                    "num_samples": 100}),
            "LIME": (ImageLime, {"num_segments": 50, "num_samples": 300})
        })
