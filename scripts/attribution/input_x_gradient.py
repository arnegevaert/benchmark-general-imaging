from captum import attr


class InputXGradient:
    def __init__(self, model):
        self.method = attr.InputXGradient(model)

    def __call__(self, x, target):
        return self.method.attribute(x, target=target)
