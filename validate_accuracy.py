import argparse
import torch
from experiments.general_imaging.lib.dataset_models import get_dataset_model
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import sklearn.metrics as metrics


def test_epoch(model, dl):
    model.to(device)
    model.eval()
    with torch.no_grad():
        top_one = []
        top_five = []

        predictions = []
        true_labels = []
        for batch, labels in tqdm(dl):
            labels = labels.to(device)
            batch = batch.to(device)
            out = model(batch)  # Get model output

            index_order = torch.argsort(out, dim=1)  # Get the indices from low to high
            # If final index matches label, top-one is correct
            top_one.append((index_order[:, -1].view(-1, 1) == labels.view(-1, 1)).any(dim=1))
            # If final 5 indices contain label, top-five is correct
            top_five.append((index_order[:, -5:] == labels.view(-1, 1)).any(dim=1))

            predictions.extend(torch.softmax(out, dim=1).cpu().numpy())
            true_labels.extend(labels.cpu().numpy())
        predictions = np.vstack(predictions)

        top_one = torch.cat(top_one)
        top_five = torch.cat(top_five)

        top_one_acc = torch.sum(top_one) / top_one.shape[0]
        top_five_acc = torch.sum(top_five) / top_five.shape[0]
        balanced_acc = metrics.balanced_accuracy_score(true_labels, predictions.argmax(axis=1))
        auc = metrics.roc_auc_score(true_labels, predictions, average="macro", multi_class='ovo', labels=np.arange(predictions.shape[1]))
    return top_one_acc.item(), top_five_acc.item(), balanced_acc, auc



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset", type=str, required=True)
    parser.add_argument("-m","--model", type=str, required=True)
    parser.add_argument("-b", "--batch-size", type=int, required=True)
    parser.add_argument("-c", "--cuda", action="store_true")

    # Parse arguments
    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() and args.cuda else "cpu"

    # Get dataset, model, methods
    ds, model, sample_shape = get_dataset_model(args.dataset, args.model)
    model.eval()
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=4)
    top_one, top_five, balanced_acc, auc = test_epoch(model, dl)
    print("validation set results:\n"
          "top-one accuracy: {:f} \n"
          "top-five accuracy: {:f} \n"
          "balanced accuracy: {:f} \n"
          "AUC: {:f}".format(top_one, top_five, balanced_acc, auc))
