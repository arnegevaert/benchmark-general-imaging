from captum import attr


class Gradient:
    def __init__(self, model):
        self.method = attr.Saliency(model)

    def __call__(self, x, target):
        return self.method.attribute(x, target=target)
