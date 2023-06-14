from attribench.metrics.result import MetricResult
from pandas import DataFrame
from typing import Dict, Tuple
import os

METRICS = {
    "impact_coverage": "Cov",
    "ms_deletion": "MSDel",
    "ms_insertion": "MSIns",
    "sens_n": "SensN",
    "seg_sens_n": "SegSensN",
    "max_sensitivity": "MaxSens",
    "deletion_morf": "Del - MoRF",
    "deletion_lerf": "Del - LeRF",
    "insertion_morf": "Ins - MoRF",
    "insertion_lerf": "Ins - LeRF",
    "irof_morf": "IROF - MoRF",
    "irof_lerf": "IROF - LeRF",
    "infidelity_square": "INFD - SQ",
    "infidelity_noisy_baseline": "INFD - BL",
}

METHODS = {
    "Gradient": "Gradient",
    "InputXGradient": "InputXGradient",
    "Deconvolution": "Deconvolution",
    "GuidedBackprop": "GuidedBackprop",
    "DeepLift": "DeepLIFT",
    "GradCAM": "GradCAM",
    "GuidedGradCAM": "GuidedGradCAM",
    "IntegratedGradients": "IntegratedGradients",
    "ExpectedGradients": "ExpectedGradients",
    "SmoothGrad": "SmoothGrad",
    "VarGrad": "VarGrad",
    "DeepShap": "DeepSHAP",
    "KernelShap": "KernelSHAP",
    "LIME": "LIME",
}


def _rename_metrics_methods(dfs):
    res = {}
    for key in dfs:
        # Get alternative metric name
        new_key = key
        for alt_key in METRICS:
            if key.startswith(alt_key):
                new_key = key.replace(alt_key, METRICS[alt_key])

        # Rename methods
        df, higher_is_better = dfs[key]
        df = df.rename(columns=METHODS)

        # Save result
        res[new_key] = (df, higher_is_better)
    return res


def _subtract_baseline(df, baseline_method):
    df = df.sub(df[baseline_method], axis=0)
    df = df.drop(columns=[baseline_method])
    return df


def _add_infidelity(dirname, baseline, result):
    infidelity_object = MetricResult.load(
        os.path.join(dirname, "infidelity.h5")
    )
    for perturbation_generator in ["square", "noisy_baseline"]:
        df, higher_is_better = infidelity_object.get_df(
            activation_fn="linear",
            perturbation_generator=perturbation_generator,
        )
        result["infidelity_" + perturbation_generator] = (
            _subtract_baseline(df, baseline) if baseline is not None else df,
            higher_is_better,
        )


def _get_default_dataframes(dirname, baseline):
    """
    Returns a dictionary of dataframes, where the keys are the metric names
    and the values are tuples of (dataframe, higher_is_better).

    Extracts the default dataframes from each metric:
    - masker = "constant"
    - activation_fn = "linear"
    """
    result: Dict[str, Tuple[DataFrame, bool]] = {}
    available_files = os.listdir(dirname)

    # Add simple metrics (masker, activation_fn)
    simple_metrics = [
        key
        for key in METRICS.keys()
        if key
        not in [
            "infidelity",
            "ms_deletion",
            "ms_insertion",
            "max_sensitivity",
            "impact_coverage",
        ]
    ]
    for metric_name in simple_metrics:
        filename = metric_name + ".h5"
        if filename in available_files:
            result_object = MetricResult.load(os.path.join(dirname, filename))
            df, higher_is_better = result_object.get_df(
                masker="constant", activation_fn="linear"
            )
            result[metric_name] = (
                _subtract_baseline(df, baseline)
                if baseline is not None
                else df,
                higher_is_better,
            )

    # Add minimal subset (masker)
    for metric_name in ["ms_deletion", "ms_insertion"]:
        filename = metric_name + ".h5"
        if filename in available_files:
            result_object = MetricResult.load(os.path.join(dirname, filename))
            df, higher_is_better = result_object.get_df(masker="constant")
            result[metric_name] = (
                _subtract_baseline(df, baseline)
                if baseline is not None
                else df,
                higher_is_better,
            )

    # Add infidelity (perturbation_generator, activation_fn)
    if "infidelity.h5" in available_files:
        _add_infidelity(dirname, baseline, result)

    # Add max-sensitivity and impact coverage (no arguments)
    for metric_name in ["max_sensitivity", "impact_coverage"]:
        filename = metric_name + ".h5"
        if filename in available_files:
            result_object = MetricResult.load(os.path.join(dirname, filename))
            df, higher_is_better = result_object.get_df()
            result[metric_name] = (
                _subtract_baseline(df, baseline)
                if baseline is not None
                else df,
                higher_is_better,
            )

    return _rename_metrics_methods(result)


def _get_all_dataframes(dirname, baseline):
    """
    Returns a dictionary of dataframes, where the keys are the metric names
    and the values are tuples of (dataframe, higher_is_better).

    Extracts an extended selection of dataframes from each metric:
    - masker = ["constant", "random", "blur"]
    - activation_fn = "linear"
    """
    result: Dict[str, Tuple[DataFrame, bool]] = {}
    available_files = os.listdir(dirname)
    maskers = ["constant", "random", "blurring"]

    # Add simple metrics (masker, activation_fn)
    simple_metrics = [
        key
        for key in METRICS.keys()
        if key
        not in [
            "infidelity",
            "ms_deletion",
            "ms_insertion",
            "max_sensitivity",
            "impact_coverage",
        ]
    ]
    for metric_name in simple_metrics:
        filename = metric_name + ".h5"
        if filename in available_files:
            result_object = MetricResult.load(os.path.join(dirname, filename))
            for masker in maskers:
                df, higher_is_better = result_object.get_df(
                    masker=masker, activation_fn="linear"
                )
                result[f"{metric_name} - {masker}"] = (
                    _subtract_baseline(df, baseline)
                    if baseline is not None
                    else df,
                    higher_is_better,
                )

    # Add minimal subset (masker)
    for metric_name in ["ms_deletion", "ms_insertion"]:
        filename = metric_name + ".h5"
        if filename in available_files:
            result_object = MetricResult.load(os.path.join(dirname, filename))
            for masker in maskers:
                df, higher_is_better = result_object.get_df(masker=masker)
                result[f"{metric_name} - {masker}"] = (
                    _subtract_baseline(df, baseline)
                    if baseline is not None
                    else df,
                    higher_is_better,
                )

    # Add infidelity (perturbation_generator, activation_fn)
    if "infidelity.h5" in available_files:
        _add_infidelity(dirname, baseline, result)

    # Add max-sensitivity and impact coverage (no arguments)
    for metric_name in ["max_sensitivity", "impact_coverage"]:
        filename = metric_name + ".h5"
        if filename in available_files:
            result_object = MetricResult.load(os.path.join(dirname, filename))
            df, higher_is_better = result_object.get_df()
            result[metric_name] = (
                _subtract_baseline(df, baseline)
                if baseline is not None
                else df,
                higher_is_better,
            )

    return _rename_metrics_methods(result)


def get_dataframes(dirname, mode="default", baseline=None):
    if mode not in ["default", "all"]:
        raise ValueError("mode must be one of ['default', 'all']")

    if mode == "default":
        return _get_default_dataframes(dirname, baseline)
    elif mode == "all":
        # return _get_all_dataframes(dirname)
        return _get_all_dataframes(dirname, baseline)
