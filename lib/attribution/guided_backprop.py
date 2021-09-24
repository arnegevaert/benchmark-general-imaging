from captum import attr


class GuidedBackprop:
    def __init__(self, model):
        self.method = attr.GuidedBackprop(model)

    def __call__(self, x, target):
        return self.method.attribute(x, target=target)
