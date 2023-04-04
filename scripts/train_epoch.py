import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm


def train_epoch(model: nn.Module, train_dl: DataLoader, test_dl: DataLoader,
                label_type: str, optimizer, criterion, device: str):
    assert label_type == "obj" or label_type == "scene"
    # Train
    prog = tqdm(train_dl, desc=f"Training")
    total_samples, correct_samples = 0, 0
    losses = []
    model.train()
    for batch, scene_labels, object_labels in prog:
        batch = batch.to(device)
        labels = scene_labels.to(device) if label_type == "scene" else object_labels.to(device)
        optimizer.zero_grad()

        y_pred = model(batch)
        loss = criterion(y_pred, labels)
        loss.backward()
        optimizer.step()

        losses.append(loss.item())
        total_samples += batch.size(0)
        correct_samples += (torch.argmax(y_pred, dim=1) == labels).sum().item()
        prog.set_postfix({"loss": sum(losses)/len(losses),
                          "acc": f"{correct_samples}/{total_samples} ({100*correct_samples/total_samples:.2f})"})

    # Test
    prog = tqdm(test_dl, desc="Testing")
    total_samples, correct_samples = 0, 0
    model.eval()
    for batch, scene_labels, object_labels in prog:
        with torch.no_grad():
            batch = batch.to(device)
            labels = scene_labels.to(device) if label_type == "scene" else object_labels.to(device)
            y_pred = model(batch)
            total_samples += batch.size(0)
            correct_samples += (torch.argmax(y_pred, dim=1) == labels).sum().item()
            prog.set_postfix(
                {"acc": f"{correct_samples}/{total_samples} ({100 * correct_samples / total_samples:.2f}%)"})
