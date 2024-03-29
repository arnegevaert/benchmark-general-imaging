from attribench.result import MetricResult
from pandas import DataFrame
from typing import Dict, Tuple, Optional
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
    "infidelity_gaussian": "INFD - GA",
    "parameter_randomization": "PR",
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
    "Random": "Random",
    "XRAI": "XRAI",
    "GradCAM++": "GradCAM++",
    "ScoreCAM": "ScoreCAM",
}


def _rename_metrics_methods(
    dfs: Dict[str, Tuple[DataFrame, bool]]
) -> Dict[str, Tuple[DataFrame, bool]]:
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


def _subtract_baseline(df: DataFrame, baseline_method: str) -> DataFrame:
    df = df.sub(df[baseline_method], axis=0)
    df = df.drop(columns=[baseline_method])
    return df


def get_metric_result(
    dirname: str, metric_name: str
) -> Optional[MetricResult]:
    csv_path = os.path.join(dirname, metric_name)
    h5_path = os.path.join(dirname, metric_name + ".h5")
    if os.path.isdir(csv_path):
        return MetricResult.load(csv_path)
    elif os.path.isfile(h5_path):
        return MetricResult.load(h5_path)
    else:
        return None


def _add_infidelity(
    dirname: str,
    result: Dict[str, Tuple[DataFrame, bool]],
    baseline: str | None = None,
    activation_fns=None,
):
    infidelity_object = get_metric_result(dirname, "infidelity")
    if infidelity_object is not None:
        perturbation_generators = ["noisy_baseline", "square"]
        for perturbation_generator in perturbation_generators:
            if (
                perturbation_generator
                in infidelity_object.levels["perturbation_generator"]
            ):
                if activation_fns is None:
                    activation_fns = infidelity_object.levels["activation_fn"]
                for activation_fn in activation_fns:
                    df, higher_is_better = infidelity_object.get_df(
                        activation_fn=activation_fn,
                        perturbation_generator=perturbation_generator,
                    )
                    result["infidelity_" + perturbation_generator] = (
                        _subtract_baseline(df, baseline)
                        if baseline is not None
                        else df,
                        higher_is_better,
                    )


def _add_masker_activation_metrics(
    dirname: str,
    result: Dict[str, Tuple[DataFrame, bool]],
    baseline: str | None = None,
    maskers=None,
    activation_fns=None,
):
    masker_activation_metrics = [
        key
        for key in METRICS.keys()
        if key
        not in [
            "infidelity",
            "ms_deletion",
            "ms_insertion",
            "max_sensitivity",
            "impact_coverage",
            "parameter_randomization"
        ]
    ]
    for metric_name in masker_activation_metrics:
        result_object = get_metric_result(dirname, metric_name)
        if result_object is not None:
            if maskers is None:
                maskers = result_object.levels["masker"]
            if activation_fns is None:
                activation_fns = result_object.levels["activation_fn"]
            for masker in maskers:
                for activation_fn in activation_fns:
                    df, higher_is_better = result_object.get_df(
                        masker=masker, activation_fn=activation_fn
                    )
                    if len(maskers) == 1 and len(activation_fns) == 1:
                        key = metric_name
                    elif len(maskers) == 1:
                        key = f"{metric_name} - {activation_fn}"
                    elif len(activation_fns) == 1:
                        key = f"{metric_name} - {masker}"
                    else:
                        key = f"{metric_name} - {masker} - {activation_fn}"
                    result[key] = (
                        _subtract_baseline(df, baseline)
                        if baseline is not None
                        else df,
                        higher_is_better,
                    )


def _add_masker_metrics(
    dirname: str,
    result: Dict[str, Tuple[DataFrame, bool]],
    baseline: str | None = None,
    maskers=None,
):
    for metric_name in ["ms_deletion", "ms_insertion"]:
        result_object = get_metric_result(dirname, metric_name)
        if result_object is not None:
            if maskers is None:
                maskers = result_object.levels["masker"]
            for masker in maskers:
                df, higher_is_better = result_object.get_df(masker=masker)
                if len(maskers) == 1:
                    key = metric_name
                else:
                    key = f"{metric_name} - {masker}"
                result[key] = (
                    _subtract_baseline(df, baseline)
                    if baseline is not None
                    else df,
                    higher_is_better,
                )


def _add_no_arg_metrics(
    dirname: str,
    result: Dict[str, Tuple[DataFrame, bool]],
    baseline: str | None = None,
):
    for metric_name in ["max_sensitivity", "impact_coverage"]:
        result_object = get_metric_result(dirname, metric_name)
        if result_object is not None:
            df, higher_is_better = result_object.get_df()
            result[metric_name] = (
                _subtract_baseline(df, baseline)
                if baseline is not None
                else df,
                higher_is_better,
            )


def _add_parameter_randomization(
    dirname: str,
    result: Dict[str, Tuple[DataFrame, bool]],
    baseline: str | None = None,
):
    result_object = get_metric_result(dirname, "parameter_randomization")
    if result_object is not None:
        df, higher_is_better = result_object.get_df()
        result["parameter_randomization"] = (
            _subtract_baseline(df, baseline)
            if baseline is not None
            else df,
            higher_is_better,
        )


def _get_default_dataframes(
    dirname: str, data_type: str, baseline: str | None = None, include_pr=False
):
    """
    Returns a dictionary of dataframes, where the keys are the metric names
    and the values are tuples of (dataframe, higher_is_better).

    Extracts the default dataframes from each metric:
    - masker = "constant" or "tabular"
    - activation_fn = "linear"
    """
    result: Dict[str, Tuple[DataFrame, bool]] = {}

    # Add masker and activation metrics (masker, activation_fn)
    _add_masker_activation_metrics(
        dirname,
        result,
        baseline,
        maskers=["constant"] if data_type == "image" else ["tabular"],
        activation_fns=["linear"],
    )

    # Add minimal subset (masker)
    _add_masker_metrics(
        dirname,
        result,
        baseline,
        maskers=["constant"] if data_type == "image" else ["tabular"],
    )

    # Add infidelity (perturbation_generator, activation_fn)
    _add_infidelity(dirname, result, baseline, activation_fns=["linear"])

    # Add max-sensitivity and impact coverage (no arguments)
    _add_no_arg_metrics(dirname, result, baseline)

    # Add parameter randomization if specified
    if include_pr:
        _add_parameter_randomization(dirname, result, baseline)

    return _rename_metrics_methods(result)


def _get_all_dataframes(
    dirname: str, baseline: str | None = None, include_pr=False
):
    """
    Returns a dictionary of dataframes, where the keys are the metric names
    and the values are tuples of (dataframe, higher_is_better).

    Extracts an extended selection of dataframes from each metric:
    - masker = ["constant", "random", "blur"]
    - activation_fn = "linear"
    """
    result: Dict[str, Tuple[DataFrame, bool]] = {}

    # Add masker and activation metrics (masker, activation_fn)
    _add_masker_activation_metrics(
        dirname,
        result,
        baseline,
        maskers=None,
        activation_fns=["linear"],
    )

    # Add minimal subset (masker)
    _add_masker_metrics(dirname, result, baseline, maskers=None)

    # Add infidelity (perturbation_generator, activation_fn)
    _add_infidelity(dirname, result, baseline, activation_fns=["linear"])

    # Add max-sensitivity and impact coverage (no arguments)
    _add_no_arg_metrics(dirname, result, baseline)

    # Add parameter randomization if specified
    if include_pr:
        _add_parameter_randomization(dirname, result, baseline)

    return _rename_metrics_methods(result)


def get_dataframes(
    dirname: str,
    mode="default",
    baseline: str | None = None,
    data_type="image",
    include_pr=False,
) -> Dict[str, Tuple[DataFrame, bool]]:
    assert data_type in ("image", "tabular")
    if mode == "default":
        return _get_default_dataframes(dirname, data_type, baseline, include_pr)
    elif mode == "all":
        # return _get_all_dataframes(dirname)
        return _get_all_dataframes(dirname, baseline, include_pr)
    else:
        raise ValueError("mode must be one of ['default', 'all']")
