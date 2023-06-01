from captum import attr
import warnings


class GuidedBackprop:
    def __init__(self, model):
        self.method = attr.GuidedBackprop(model)

    def __call__(self, x, target):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return self.method.attribute(x, target=target)
