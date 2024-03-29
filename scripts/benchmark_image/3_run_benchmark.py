import torch
import argparse
import warnings
from util.datasets import ALL_DATASETS, get_dataset
from util.attribution.method_factory import get_method_factory
from util.models import ModelFactoryImpl
from attribench.data import AttributionsDataset, HDF5Dataset
from attribench.distributed.metrics import (
    Deletion,
    Insertion,
    Irof,
    Infidelity,
    SensitivityN,
    MinimalSubset,
    MaxSensitivity,
    ImpactCoverage,
    ParameterRandomization,
)
from attribench.masking.image import ConstantImageMasker, RandomImageMasker, BlurringImageMasker
from attribench.functional.metrics.infidelity import (
    NoisyBaselinePerturbationGenerator,
    GaussianPerturbationGenerator,
    SquarePerturbationGenerator,
)
import os


def remove_if_present(filenames):
    for filename in os.listdir(args.output_dir):
        if filename in filenames:
            os.remove(os.path.join(args.output_dir, filename))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset", type=str, default="MNIST", choices=ALL_DATASETS
    )
    parser.add_argument(
        "--samples-file", type=str, default="out/results/MNIST/samples.h5"
    )
    parser.add_argument(
        "--attrs-file", type=str, default="out/results/MNIST/attributions.h5"
    )
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--model", type=str, default="BasicCNN")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--output-dir", type=str, default="out")
    parser.add_argument("--patch-folder", type=str)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--metrics",
        nargs="*",
        default=[
            "deletion",
            "insertion",
            "irof",
            "infidelity",
            "sensitivity_n",
            "seg_sensitivity_n",
            "minimal_subset_deletion",
            "minimal_subset_insertion",
            "max_sensitivity",
            "impact_coverage",
            "parameter_randomization"
        ],
    )
    args = parser.parse_args()

    # Initialize dataset, model, hyperparameters
    samples_dataset = HDF5Dataset(args.samples_file)
    reference_dataset = get_dataset(args.dataset, args.data_dir)
    attributions_dataset = AttributionsDataset(
        samples=samples_dataset,
        path=args.attrs_file,
        aggregate_dim=0,
        aggregate_method="mean",
    )
    model_factory = ModelFactoryImpl(args.dataset, args.data_dir, args.model)
    maskers = {
        "constant": ConstantImageMasker(masking_level="pixel"),
        "random": RandomImageMasker(masking_level="pixel"),
        "blurring": BlurringImageMasker(masking_level="pixel", kernel_size=0.5),
    }
    method_factory = get_method_factory(
        args.batch_size,
        reference_dataset=reference_dataset,
        methods=attributions_dataset.method_names
    )
    activation_fns = ["linear", "softmax"]

    # Check if output directory exists and is empty
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    else:
        if os.listdir(args.output_dir) and not args.overwrite:
            raise ValueError(
                "Output directory is not empty. "
                "Use --overwrite to overwrite existing files."
            )

    # Check if args are consistent
    if "impact_coverage" in args.metrics and args.patch_folder is None:
        warnings.warn("Patch folder not set, skipping impact coverage.")
        args.metrics.remove("impact_coverage")
    
    ############
    # DELETION #
    ############
    if "deletion" in args.metrics:
        if args.overwrite:
            remove_if_present(["deletion_morf.h5", "deletion_lerf.h5"])
        for mode in ["morf", "lerf"]:
            print(f"Running deletion-{mode}...")
            deletion = Deletion(
                model_factory,
                attributions_dataset,
                args.batch_size,
                maskers=maskers,
                activation_fns=activation_fns,
                mode=mode,
                start=0.0,
                stop=0.15,
                num_steps=100,
            )
            deletion_output_file = os.path.join(
                args.output_dir, f"deletion_{mode}.h5"
            )
            deletion.run(result_path=deletion_output_file)
            print()

    #############
    # INSERTION #
    #############
    if "insertion" in args.metrics:
        if args.overwrite:
            remove_if_present(["insertion_morf.h5", "insertion_lerf.h5"])
        for mode in ["morf", "lerf"]:
            print(f"Running insertion-{mode}...")
            insertion = Insertion(
                model_factory,
                attributions_dataset,
                args.batch_size,
                maskers=maskers,
                activation_fns=activation_fns,
                mode=mode,
                start=0.0,
                stop=0.15,
                num_steps=100,
            )
            insertion_output_file = os.path.join(
                args.output_dir, f"insertion_{mode}.h5"
            )
            insertion.run(result_path=insertion_output_file)
            print()

    ########
    # IROF #
    ########
    if "irof" in args.metrics:
        if args.overwrite:
            remove_if_present(["irof_morf.h5", "irof_lerf.h5"])
        for mode in ["morf", "lerf"]:
            print(f"Running IROF-{mode}...")
            irof = Irof(
                model_factory,
                attributions_dataset,
                args.batch_size,
                maskers=maskers,
                activation_fns=activation_fns,
                mode=mode,
                start=0.0,
                stop=1.0,
                num_steps=100,
            )
            irof_output_file = os.path.join(args.output_dir, f"irof_{mode}.h5")
            irof.run(result_path=irof_output_file)
            print()

    ##############
    # INFIDELITY #
    ##############
    if "infidelity" in args.metrics:
        if args.overwrite:
            remove_if_present(["infidelity.h5"])
        print(f"Running Infidelity...")
        perturbation_generators = {
            "noisy_baseline": NoisyBaselinePerturbationGenerator(sd=0.2),
            "gaussian": GaussianPerturbationGenerator(sd=0.2),
            "square": SquarePerturbationGenerator(square_size=5),
        }

        infidelity = Infidelity(
            model_factory,
            attributions_dataset,
            args.batch_size,
            activation_fns=activation_fns,
            perturbation_generators=perturbation_generators,
            num_perturbations=1000,
        )
        infidelity_output_file = os.path.join(args.output_dir, "infidelity.h5")
        infidelity.run(result_path=infidelity_output_file)
        print()

    #################
    # SENSITIVITY-N #
    #################
    if "sensitivity_n" in args.metrics:
        if args.overwrite:
            remove_if_present(["sens_n.h5"])
        print("Running Sensitivity-N...")
        sens_n = SensitivityN(
            model_factory,
            attributions_dataset,
            args.batch_size,
            maskers=maskers,
            activation_fns=activation_fns,
            min_subset_size=0.1,
            max_subset_size=0.5,
            num_steps=10,
            num_subsets=100,
            segmented=False,
        )
        sens_n_output_file = os.path.join(args.output_dir, "sens_n.h5")
        sens_n.run(result_path=sens_n_output_file)
        print()

    #####################
    # SEG-SENSITIVITY-N #
    #####################
    if "seg_sensitivity_n" in args.metrics:
        if args.overwrite:
            remove_if_present(["seg_sens_n.h5"])
        print("Running Seg-Sensitivity-N...")
        seg_sens_n = SensitivityN(
            model_factory,
            attributions_dataset,
            args.batch_size,
            maskers=maskers,
            activation_fns=activation_fns,
            min_subset_size=0.1,
            max_subset_size=0.5,
            num_steps=10,
            num_subsets=100,
            segmented=True,
        )
        seg_sens_n_output_file = os.path.join(args.output_dir, "seg_sens_n.h5")
        seg_sens_n.run(result_path=seg_sens_n_output_file)
        print()

    ###########################
    # MINIMAL SUBSET DELETION #
    ###########################
    if "minimal_subset_deletion" in args.metrics:
        if args.overwrite:
            remove_if_present(["ms_deletion.h5"])
        print("Running Minimal Subset Deletion...")
        ms_deletion = MinimalSubset(
            model_factory,
            attributions_dataset,
            args.batch_size,
            maskers=maskers,
            mode="deletion",
        )
        ms_deletion_output_file = os.path.join(
            args.output_dir, "ms_deletion.h5"
        )
        ms_deletion.run(result_path=ms_deletion_output_file)
        print()

    ############################
    # MINIMAL SUBSET INSERTION #
    ############################
    if "minimal_subset_insertion" in args.metrics:
        if args.overwrite:
            remove_if_present(["ms_insertion.h5"])
        print("Running Minimal Subset Insertion...")
        ms_insertion = MinimalSubset(
            model_factory,
            attributions_dataset,
            args.batch_size,
            maskers=maskers,
            mode="insertion",
        )
        ms_insertion_output_file = os.path.join(
            args.output_dir, "ms_insertion.h5"
        )
        ms_insertion.run(result_path=ms_insertion_output_file)
        print()

    ###################
    # MAX-SENSITIVITY #
    ###################
    if "max_sensitivity" in args.metrics:
        if args.overwrite:
            remove_if_present(["max_sensitivity.h5"])
        print("Running Max-Sensitivity...")
        max_sensitivity = MaxSensitivity(
            model_factory,
            attributions_dataset,
            args.batch_size,
            method_factory,
            num_perturbations=50,
            radius=0.1,
        )
        max_sensitivity_output_file = os.path.join(
            args.output_dir, "max_sensitivity.h5"
        )
        max_sensitivity.run(result_path=max_sensitivity_output_file)
        print()

    ###################
    # IMPACT COVERAGE #
    ###################
    if "impact_coverage" in args.metrics:
        if args.overwrite:
            remove_if_present(["impact_coverage.h5"])
        print("Running Impact Coverage...")
        coverage = ImpactCoverage(
            model_factory,
            samples_dataset,
            args.batch_size,
            method_factory,
            args.patch_folder,
        )
        coverage_output_file = os.path.join(
            args.output_dir, "impact_coverage.h5"
        )
        coverage.run(result_path=coverage_output_file)
        print()

    ###########################
    # PARAMETER RANDOMIZATION #
    ###########################
    if "parameter_randomization" in args.metrics:
        if args.overwrite:
            remove_if_present(["parameter_randomization.h5"])
        print("Running Parameter Randomization...")
        parameter_randomization = ParameterRandomization(
            model_factory,
            attributions_dataset,
            args.batch_size,
            method_factory,
        )
        parameter_randomization_output_file = os.path.join(
            args.output_dir, "parameter_randomization.h5"
        )
        parameter_randomization.run(result_path=parameter_randomization_output_file)
        print()
