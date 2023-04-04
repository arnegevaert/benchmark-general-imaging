from captum import attr


class IntegratedGradients:
    def __init__(self, model, internal_batch_size, num_steps):
        self.method = attr.IntegratedGradients(model)
        self.internal_batch_size = internal_batch_size
        self.num_steps = num_steps

    def __call__(self, x, target):
        return self.method.attribute(x, target=target,
                                     internal_batch_size=self.internal_batch_size, n_steps=self.num_steps)
