from attrbench.metrics.result import MetricResult
from pandas import DataFrame
from typing import Dict, Tuple
import os

METRICS = {
    "deletion_morf": "Del - MoRF",
    "insertion_morf": "Ins - MoRF",
    "deletion_lerf": "Del - LeRF",
    "insertion_lerf": "Ins - LeRF",
    "ms_insertion": "MSIns",
    "ms_deletion": "MSDel",
    "irof_morf": "IROF - MoRF",
    "irof_lerf": "IROF - LeRF",
    "sens_n": "SensN",
    "seg_sens_n": "SegSensN",
    "infidelity": "INFD",
    "max_sensitivity": "MaxSens",
    "impact_coverage": "Cov",
}


def get_dataframes(dirname, mode="default"):
    if mode not in ["default", "all"]:
        raise ValueError("mode must be one of ['default', 'all']")

    if mode == "default":
        return _get_default_dataframes(dirname)
    elif mode == "all":
        #return _get_all_dataframes(dirname)
        return _get_default_dataframes(dirname)


def _get_default_dataframes(dirname):
    """
    Returns a dictionary of dataframes, where the keys are the metric names
    and the values are tuples of (dataframe, higher_is_better).

    Extracts the default dataframes from each metric:
    - masker = "constant"
    - activation_fn = "linear"
    """
    result: Dict[str, Tuple[DataFrame, bool]] = {}

    # Add simple metrics (masker, activation_fn)
    simple_metrics = [key for key in METRICS.keys() if key != "infidelity"]
    for metric_name in simple_metrics:
        filename = metric_name + ".h5"
        result_object = MetricResult.load(os.path.join(dirname, filename))
        df, higher_is_better = result_object.get_df(
            masker="constant", activation_fn="linear"
        )
        result[METRICS[metric_name]] = (df, higher_is_better)

    # Add infidelity
    infidelity_object = MetricResult.load(
        os.path.join(dirname, "infidelity.h5")
    )
    for perturbation_generator, abbrev in [
        ("square", "SQ"),
        ("noisy_bl", "BL"),
    ]:
        df, higher_is_better = infidelity_object.get_df(
            activation_fn="linear",
            perturbation_generator=perturbation_generator,
        )
        metric_name = "INFD - " + abbrev
        result[metric_name] = (df, higher_is_better)
    
    return result


def _get_all_dataframes(dirname):
    pass
