from attrbench.data import AttributionsDataset, HDF5Dataset
from attrbench.metrics.infidelity import Infidelity,\
        GaussianPerturbationGenerator, NoisyBaselinePerturbationGenerator,\
        SquarePerturbationGenerator
from .util import arguments


if __name__ == "__main__":
    parser = arguments.get_default_argparser()
    args = parser.parse_args()

    samples_dataset = HDF5Dataset(args.samples_dataset)
    attributions_dataset = AttributionsDataset(
            samples_dataset, args.attributions_dataset,
            aggregate_axis=0, aggregate_method="mean",
            group_attributions=True)
    perturbation_generators = {
            "gaussian": GaussianPerturbationGenerator(sd=0.2),
            "noisy_bl": NoisyBaselinePerturbationGenerator(sd=0.2),
            "square": SquarePerturbationGenerator(square_size=5)
            }

