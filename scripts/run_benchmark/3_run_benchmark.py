import argparse
from util.datasets import ALL_DATASETS, get_dataset
from util.attribution.method_factory import get_method_factory
from util.models import get_model
from attrbench.data import AttributionsDataset, HDF5Dataset, IndexDataset
from attrbench.metrics import Deletion, Irof, Infidelity, SensitivityN,\
    MinimalSubset, MaxSensitivity, ImpactCoverage
from attrbench.distributed import Model
from attrbench.masking import ConstantMasker, RandomMasker, BlurringMasker
from attrbench.metrics.infidelity import NoisyBaselinePerturbationGenerator,\
    GaussianPerturbationGenerator, SquarePerturbationGenerator
import os


def remove_if_present(filenames):
    for filename in os.listdir(args.output_dir):
        if filename in filenames:
            os.remove(os.path.join(args.output_dir, filename))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="MNIST",
                        choices=ALL_DATASETS)
    parser.add_argument("--samples-file", type=str, default="samples.h5")
    parser.add_argument("--attrs-file", type=str, default="attributions.h5")
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--model", type=str, default="BasicCNN")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--output-dir", type=str, default="out")
    parser.add_argument("--include-ic", action="store_true")
    parser.add_argument("--patch-folder", type=str)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    # Initialize dataset, model, hyperparameters
    samples_dataset = HDF5Dataset(args.samples_file)
    reference_dataset = get_dataset(args.dataset, args.data_dir)
    attributions_dataset = AttributionsDataset(samples_dataset, 
                                  args.attrs_file,
                                  aggregate_axis=0, aggregate_method="mean")
    model = Model(get_model(args.dataset, args.data_dir, args.model))
    maskers = {
        "constant": ConstantMasker(feature_level="pixel"),
        "random": RandomMasker(feature_level="pixel"),
        "blurring": BlurringMasker(feature_level="pixel",
                                   kernel_size=0.5)
        }
    method_factory = get_method_factory(args.batch_size,
                                        reference_dataset=reference_dataset)
    activation_fns = ["linear", "softmax"]

    # Check if output directory exists and is empty
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    else:
        if os.listdir(args.output_dir) and not args.overwrite:
            raise ValueError("Output directory is not empty. "
                             "Use --overwrite to overwrite existing files.")
    
    # Check if args are consistent
    if args.include_ic and args.patch_folder is None:
        raise ValueError("If --include-ic is set, --patch-folder must be set.")
    
    ############
    # DELETION #
    ############
    if args.overwrite:
        remove_if_present(["deletion_morf.h5", "deletion_lerf.h5"])
    for mode in ["morf", "lerf"]:
        print(f"Running deletion-{mode}...")
        deletion = Deletion(model, attributions_dataset, args.batch_size,
                            maskers=maskers, activation_fns=activation_fns,
                            mode=mode, start=0., stop=0.15,
                            num_steps=100)
        deletion_output_file = os.path.join(args.output_dir, 
                                            f"deletion_{mode}.h5")
        deletion.run(result_path=deletion_output_file)
        print()

    #############
    # INSERTION #
    #############
    if args.overwrite:
        remove_if_present(["insertion_morf.h5", "insertion_lerf.h5"])
    for mode in ["morf", "lerf"]:
        print(f"Running insertion-{mode}...")
        insertion = Deletion(model, attributions_dataset, args.batch_size,
                            maskers=maskers, activation_fns=activation_fns,
                            mode=mode, start=1., stop=0.85,
                            num_steps=100)
        insertion_output_file = os.path.join(args.output_dir, 
                                            f"insertion_{mode}.h5")
        insertion.run(result_path=insertion_output_file)
        print()

    ########
    # IROF #
    ########
    if args.overwrite:
        remove_if_present(["irof_morf.h5", "irof_lerf.h5"])
    for mode in ["morf", "lerf"]:
        print(f"Running IROF-{mode}...")
        irof = Irof(model, attributions_dataset, args.batch_size,
                    maskers=maskers, activation_fns=activation_fns,
                    start=0., stop=1., num_steps=100)
        irof_output_file = os.path.join(args.output_dir, f"irof_{mode}.h5")
        irof.run(result_path=irof_output_file)
        print()

    ##############
    # INFIDELITY #
    ##############
    if args.overwrite:
        remove_if_present(["infidelity.h5"])
    print(f"Running Infidelity...")
    perturbation_generators = {
        "noisy_baseline": NoisyBaselinePerturbationGenerator(sd=0.2),
        "gaussian": GaussianPerturbationGenerator(sd=0.2),
        "square": SquarePerturbationGenerator(square_size=5)
    }

    attributions_dataset.group_attributions = True
    infidelity = Infidelity(model, attributions_dataset, args.batch_size,
                            perturbation_generators=perturbation_generators,
                            num_perturbations=1000,
                            activation_fns=activation_fns)
    infidelity_output_file = os.path.join(args.output_dir, "infidelity.h5")
    infidelity.run(result_path=infidelity_output_file)
    attributions_dataset.group_attributions = False
    print()

    #################
    # SENSITIVITY-N #
    #################
    if args.overwrite:
        remove_if_present(["sens_n.h5"])
    print("Running Sensitivity-N...")
    attributions_dataset.group_attributions = True
    sens_n = SensitivityN(model, attributions_dataset, args.batch_size,
                        min_subset_size=0.1, max_subset_size=0.5,
                        num_steps=10, num_subsets=100,
                        maskers=maskers, activation_fns=activation_fns)
    sens_n_output_file = os.path.join(args.output_dir, "sens_n.h5")
    sens_n.run(result_path=sens_n_output_file)
    attributions_dataset.group_attributions = False
    print()

    #####################
    # SEG-SENSITIVITY-N #
    #####################
    if args.overwrite:
        remove_if_present(["seg_sens_n.h5"])
    print("Running Seg-Sensitivity-N...")
    attributions_dataset.group_attributions = True
    seg_sens_n = SensitivityN(model, attributions_dataset, args.batch_size,
                        min_subset_size=0.1, max_subset_size=0.5,
                        num_steps=10, num_subsets=100, segmented=True,
                        maskers=maskers, activation_fns=activation_fns)
    seg_sens_n_output_file = os.path.join(args.output_dir, "seg_sens_n.h5")
    seg_sens_n.run(result_path=seg_sens_n_output_file)
    attributions_dataset.group_attributions = False
    print()

    ###########################
    # MINIMAL SUBSET DELETION #
    ###########################
    if args.overwrite:
        remove_if_present(["ms_deletion.h5"])
    print("Running Minimal Subset Deletion...")
    ms_deletion = MinimalSubset(model, attributions_dataset, args.batch_size,
                                maskers=maskers, mode="deletion")
    ms_deletion_output_file = os.path.join(args.output_dir, "ms_deletion.h5")
    ms_deletion.run(result_path=ms_deletion_output_file)
    print()
    
    ############################
    # MINIMAL SUBSET INSERTION #
    ############################
    if args.overwrite:
        remove_if_present(["ms_insertion.h5"])
    print("Running Minimal Subset Insertion...")
    ms_insertion = MinimalSubset(model, attributions_dataset, args.batch_size,
                                maskers=maskers, mode="insertion")
    ms_insertion_output_file = os.path.join(args.output_dir, "ms_insertion.h5")
    ms_insertion.run(result_path=ms_insertion_output_file)
    print()

    ###################
    # MAX-SENSITIVITY #
    ###################
    if args.overwrite:
        remove_if_present(["max_sensitivity.h5"])
    print("Running Max-Sensitivity...")
    index_dataset = IndexDataset(samples_dataset)
    max_sensitivity = MaxSensitivity(model, index_dataset,
                                     args.batch_size, method_factory,
                                     num_perturbations=50,
                                     radius=0.1)
    max_sensitivity_output_file = os.path.join(args.output_dir,
                                               "max_sensitivity.h5")
    max_sensitivity.run(result_path=max_sensitivity_output_file)
    print()

    ###################
    # IMPACT COVERAGE #
    ###################
    if args.include_ic:
        if args.overwrite:
            remove_if_present(["impact_coverage.h5"])
        print("Running Impact Coverage...")
        coverage = ImpactCoverage(model, index_dataset, args.batch_size,
                                  method_factory, args.patch_folder)
        coverage_output_file = os.path.join(args.output_dir,
                                            "impact_coverage.h5")
        coverage.run(result_path=coverage_output_file)
        print()
