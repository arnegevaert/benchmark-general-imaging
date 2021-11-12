import argparse
from tqdm import tqdm
import pandas as pd
import torch
from torch.utils.data import DataLoader
from lib import attribution
from lib.dataset_models import get_dataset_model
from attrbench.metrics import sensitivity_n, seg_sensitivity_n
from attrbench.lib.masking import ConstantMasker


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset", type=str, default="MNIST")
    parser.add_argument("-m", "--model", type=str, default="CNN")
    parser.add_argument("-b", "--batch_size", type=int, default=4)
    parser.add_argument("-n", "--num_samples", type=int, default=4)
    parser.add_argument("-i", "--iterations", type=int, default=2)
    parser.add_argument("-o", "--out_file", type=str, default="out.csv")
    parser.add_argument("-c", "--cuda", action="store_true")
    args = parser.parse_args()
    device = "cuda"  # if torch.cuda.is_available() and args.cuda else "cpu"

    ds, model, _ = get_dataset_model(args.dataset, model_name=args.model)
    dl = DataLoader(ds, batch_size=args.batch_size,
                    shuffle=True, num_workers=4)
    deepshap = attribution.DeepShap(
        model, reference_dataset=ds, n_baseline_samples=10)

    model.eval()
    model = model.to(device)

    samples_done = 0
    it = iter(dl)
    sens_n_variances = []
    seg_sens_n_variances = []
    prog = tqdm(total=args.num_samples)
    prog.update(n=0)
    while samples_done < args.num_samples:
        # Get a batch of samples
        samples, labels = next(it)
        samples = samples.to(device)
        labels = labels.to(device)

        sens_n_results = []
        seg_sens_n_results = []

        with torch.no_grad():
            # Use only correctly classified samples
            out = model(samples)
            pred = torch.argmax(out, dim=1)
            samples = samples[pred == labels]
            labels = labels[pred == labels]
            if samples.shape[0] > 0:
                attrs = deepshap(samples, labels).mean(dim=1, keepdim=True).cpu().detach().numpy()
                masker = ConstantMasker("pixel")

                for i in range(args.iterations):
                    sens_n_results.append(sensitivity_n(samples, labels, model, attrs,
                                                        min_subset_size=.1, max_subset_size=.5,
                                                        num_steps=10, num_subsets=100,
                                                        masker=masker)["linear"])
                    seg_sens_n_results.append(seg_sensitivity_n(samples, labels, model, attrs,
                                                                min_subset_size=.1, max_subset_size=.5,
                                                                num_steps=10, num_subsets=100,
                                                                masker=masker)["linear"])

                sens_n_variances.append(torch.stack(sens_n_results, dim=1).mean(dim=-1).var(dim=-1))
                seg_sens_n_variances.append(torch.stack(seg_sens_n_results, dim=1).mean(dim=-1).var(dim=-1))
        samples_done += samples.shape[0]
        prog.update(n=samples.shape[0])
    
    prog.close()
    sens_n_variances = torch.cat(sens_n_variances).numpy()
    seg_sens_n_variances = torch.cat(seg_sens_n_variances).numpy()

    df = pd.DataFrame({"sens_n": sens_n_variances, "seg_sens_n": seg_sens_n_variances})
    df.to_csv(args.out_file, index=False)