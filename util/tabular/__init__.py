from .basic_nn import BasicNN
from .openml_dataset import OpenMLDataset
from .get_method_dict import get_method_dict

_DATASETS = {
    "satimage": {"data_id": 182, "pred_type": "classification"},
    "adult": {"data_id": 1590, "pred_type": "classification"},
    "spambase": {"data_id": 44, "pred_type": "classification"},
    "dna": {"data_id": 40670, "pred_type": "classification"},
}