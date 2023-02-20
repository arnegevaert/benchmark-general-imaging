from captum import attr


class DeepLift:
    def __init__(self, model):
        self.method = attr.DeepLift(model)

    def __call__(self, x, target):
        return self.method.attribute(x, target=target)
