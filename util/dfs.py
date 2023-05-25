from attrbench.suite import SuiteResult
import numpy as np

METRICS = {
    "deletion_morf": "Del - MoRF",
    "insertion_morf": "Ins - MoRF",
    "deletion_lerf": "Del - LeRF",
    "insertion_lerf": "Ins - LeRF",
    "minimal_subset_insertion": "MSIns",
    "minimal_subset_deletion": "MSDel",
    "irof_morf": "IROF - MoRF",
    "irof_lerf": "IROF - LeRF",
    "sensitivity_n": "SensN",
    "seg_sensitivity_n": "SegSensN",
    "infidelity_square": "INFD - SQ",
    "infidelity_noisy_bl": "INFD - BL",
    "max_sensitivity": "MaxSens",
    "impact_coverage": "Cov",
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


def _translate(dfs):
    res = {}
    for key in dfs:
        df, inverted = dfs[key]
        new_key = key
        for alt_key in METRICS:
            if key.startswith(alt_key):
                new_key = key.replace(alt_key, METRICS[alt_key])
        df = df.rename(columns=METHODS)
        res[new_key] = (df, inverted)
    return res


def get_default_dfs(
    path: str, mode: str, activation_fn="linear", masker="constant"
):
    pass


def get_default_dfs_old(
    res_obj: SuiteResult,
    mode: str,
    activation_fn="linear",
    masker="constant",
    include_baseline=False,
):
    # Add simple metrics
    res = {
        metric_name: res_obj.metric_results[metric_name].get_df(
            mode=mode,
            activation_fn=activation_fn,
            masker=masker,
            include_baseline=include_baseline,
        )
        for metric_name in [
            "impact_coverage",
            "minimal_subset_deletion",
            "minimal_subset_insertion",
            "sensitivity_n",
            "seg_sensitivity_n",
            "max_sensitivity",
            "deletion_morf",
            "deletion_lerf",
            "insertion_morf",
            "insertion_lerf",
            "irof_morf",
            "irof_lerf",
        ]
        if metric_name in res_obj.metric_results.keys()
    }

    for infid_type in ("square", "noisy_bl"):
        res[f"infidelity_{infid_type}"] = res_obj.metric_results[
            "infidelity"
        ].get_df(
            mode=mode,
            perturbation_generator=infid_type,
            activation_fn=activation_fn,
            include_baseline=include_baseline,
        )
    return _translate(res)


def get_all_dfs(res_obj: SuiteResult, mode: str):
    res = dict()
    activation_fn = "linear"
    if "impact_coverage" in res_obj.metric_results.keys():
        res["impact_coverage"] = res_obj.metric_results[
            "impact_coverage"
        ].get_df(mode=mode)
    res["max_sensitivity"] = res_obj.metric_results["max_sensitivity"].get_df(
        mode=mode
    )
    for metric_name in [
        "deletion_morf",
        "deletion_lerf",
        "insertion_morf",
        "insertion_lerf",
        "irof_morf",
        "irof_lerf",
        "sensitivity_n",
        "seg_sensitivity_n",
    ]:
        for masker in ["blur", "constant", "random"]:
            if metric_name not in ["sensitivity_n", "seg_sensitivity_n"]:
                res[f"{metric_name} - {masker}"] = res_obj.metric_results[
                    metric_name
                ].get_df(
                    masker=masker,
                    activation_fn=activation_fn,
                    mode=mode,
                    normalize=True,
                )
            else:
                res[f"{metric_name} - {masker}"] = res_obj.metric_results[
                    metric_name
                ].get_df(masker=masker, activation_fn=activation_fn, mode=mode)

    for infid_type in ["square", "noisy_bl"]:
        res[f"infidelity - {infid_type}"] = res_obj.metric_results[
            "infidelity"
        ].get_df(
            perturbation_generator=infid_type,
            activation_fn=activation_fn,
            mode=mode,
        )

    for metric_name in ["minimal_subset_insertion", "minimal_subset_deletion"]:
        for masker in ["blur", "constant", "random"]:
            res[f"{metric_name} - {masker}"] = res_obj.metric_results[
                metric_name
            ].get_df(masker=masker, mode=mode)
    return _translate(res)
