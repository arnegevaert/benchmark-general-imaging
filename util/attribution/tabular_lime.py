from captum import attr
import torch
from attribench import AttributionMethod


class TabularLime(AttributionMethod):
    def __init__(self, model, num_samples):
        super().__init__(model)
        self.method = None
        self.num_samples = num_samples

    def __call__(self, batch_x: torch.Tensor, batch_target: torch.Tensor):
        # Kernel width depends on the number of features: 
        # 0.75 * sqrt(n_features)
        # This means that the LIME object has to be initialized in the first
        # call to __call__.
        # By default, LIME for tabular data uses the euclidean distance.

        if self.method is None:
            sim_fn = attr._core.lime.get_exp_kernel_similarity_function(
                kernel_width=0.75 * batch_x.shape[1] ** 0.5,
                distance_mode="euclidean"
            )
            self.method = attr.Lime(self.model, similarity_func=sim_fn)

        return torch.cat(
            [
                self.method.attribute(
                    batch_x[i, ...].unsqueeze(0),
                    target=batch_target[i],
                    n_samples=self.num_samples,
                    perturbations_per_eval=batch_x.shape[0],
                )
                for i in range(batch_x.shape[0])
            ]
        )