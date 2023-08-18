from attribench import AttributionMethod
import os
import torch
import saliency.core as saliency
import numpy as np
from .integrated_gradients import IntegratedGradients
from numpy import typing as npt
import joblib


class XRAI(AttributionMethod):
    def __init__(self, model: torch.nn.Module, batch_size: int) -> None:
        super().__init__(model)
        self.batch_size = batch_size
        self.xrai_object = saliency.XRAI()
        self.ig_object = IntegratedGradients(
            model, internal_batch_size=batch_size, num_steps=25
        )

    def __call__(
        self, batch_x: torch.Tensor, batch_target: torch.Tensor
    ) -> torch.Tensor:
        ig_attrs = self.ig_object(batch_x, batch_target).cpu().detach().numpy()

        batch_x_np: npt.NDArray = batch_x.detach().cpu().numpy()
        channels = batch_x.shape[1]

        orig_warning_setting = os.environ.get("PYTHONWARNINGS", "")
        os.environ['PYTHONWARNINGS']='ignore::FutureWarning'
        res_list = joblib.Parallel(n_jobs=-1, verbose=1)(
            joblib.delayed(self.xrai_object.GetMask)(
                im.transpose(1, 2, 0),
                None,
                None,
                batch_size=self.batch_size,
                base_attribution=ig_attrs[idx].transpose(1, 2, 0),
            )
            for idx, im in enumerate(batch_x_np)
        )
        os.environ["PYTHONWARNINGS"] = orig_warning_setting

        assert res_list is not None
        result = np.stack(res_list)
        # expand attribtions to have channels for compatibility reasons
        result = np.tile(np.expand_dims(result, axis=1), (1, channels, 1, 1))
        result = torch.from_numpy(result)
        return result
