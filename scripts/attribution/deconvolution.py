from captum import attr


class Deconvolution:
    def __init__(self, model):
        self.method = attr.Deconvolution(model)

    def __call__(self, x, target):
        return self.method.attribute(x, target=target)
