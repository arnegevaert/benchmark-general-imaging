from captum import attr


class SmoothGrad:
    def __init__(self, model, num_samples, stdev):
        self.method = attr.NoiseTunnel(attr.Saliency(model))
        self.num_samples = num_samples
        self.stdev = stdev

    def __call__(self, x, target):
        # sigma = self.noise_level / (x.max()-x.min()) # follows paper more closely, but not perfectly.
        # in paper the sigma is set per image, here per batch
        return self.method.attribute(x, target=target, nt_type="smoothgrad",
                                     nt_samples=self.num_samples,
                                     nt_samples_batch_size=1,
                                     stdevs=self.stdev)
