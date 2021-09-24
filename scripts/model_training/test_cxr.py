import argparse
from torch.utils.data import DataLoader
import sklearn.metrics
from experiments.medical_imaging.dataset_models import get_dataset_model
import numpy as np
import torch
from os import path
import os


def evaluate(gts, probabilities, pathologies, use_only_index = None):
    assert(np.all(probabilities >= 0) == True)
    assert(np.all(probabilities <= 1) == True)

    def compute_metrics_for_class(i):
         p, r, t = sklearn.metrics.precision_recall_curve(gts[:, i], probabilities[:, i])
         PR_AUC = sklearn.metrics.auc(r, p)
         ROC_AUC = sklearn.metrics.roc_auc_score(gts[:, i], probabilities[:, i])
         F1 = sklearn.metrics.f1_score(gts[:, i], preds[:, i])
         acc = sklearn.metrics.accuracy_score(gts[:, i], preds[:, i])
         count = np.sum(gts[:, i])
         return PR_AUC, ROC_AUC, F1, acc, count

    PR_AUCs = []
    ROC_AUCs = []
    F1s = []
    accs = []
    counts = []
    preds = probabilities >= 0.5

    classes = [use_only_index] if use_only_index is not None else range(len(gts[0]))

    for i in classes:
        try:
            PR_AUC, ROC_AUC, F1, acc, count = compute_metrics_for_class(i)
        except ValueError:
            continue
        PR_AUCs.append(PR_AUC)
        ROC_AUCs.append(ROC_AUC)
        F1s.append(F1)
        accs.append(acc)
        counts.append(count)
        print('Class: {!s} Count: {:d} PR AUC: {:.4f} ROC AUC: {:.4f} F1: {:.3f} Acc: {:.3f}'.format(pathologies[i], count.astype(int), PR_AUC, ROC_AUC, F1, acc))

    avg_PR_AUC = np.average(PR_AUCs)
    avg_ROC_AUC = np.average(ROC_AUCs, weights=counts)
    avg_F1 = np.average(F1s, weights=counts)

    print('Avg PR AUC: {:.3f}'.format(avg_PR_AUC))
    print('Avg ROC AUC: {:.3f}'.format(avg_ROC_AUC))
    print('Avg F1: {:.3f}'.format(avg_F1))
    return avg_PR_AUC, avg_ROC_AUC, avg_F1

def test_epoch(model, loader, criterion, epoch=1):
    """
    Returns: (AUC, ROC AUC, F1, validation loss)
    """
    model.eval()
    test_losses = []
    outs = []
    gts = []
    with torch.no_grad():
        for data in loader:
            for gt in data[1].numpy().tolist():
                gts.append(gt)
            inputs, labels = data[0].cuda(),data[1].cuda()
            outputs = model(inputs)
            loss = criterion(outputs, labels, epoch=epoch)
            test_losses.append(loss.item())
            out = torch.sigmoid(outputs).data.cpu().numpy()
            outs.extend(out)
    avg_loss = np.mean(test_losses)
    print("Validation Loss: {:.6f}".format(avg_loss))
    outs = np.array(outs)
    gts = np.array(gts)
    return evaluate(gts, outs, loader.dataset.pathologies) + (avg_loss,)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset", type=str, required=True)

    parser.add_argument("-b", "--batch-size", type=int, required=True)
    parser.add_argument("-c", "--cuda", action="store_true")

    parser.add_argument("--num-workers", type=int, default=4)


    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() and args.cuda else "cpu"
    dataset, model, sample_shape = get_dataset_model(args.dataset)
    dataset.use_gpu = device=="cuda"
    loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=args.num_workers)
    model.to(device)
    model.eval()
    criterion = dataset.weighted_loss

    _,_,_,avg_loss=test_epoch(model,loader,criterion)
    print("Validation Loss: {:.6f}".format(avg_loss))
