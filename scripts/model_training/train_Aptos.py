import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from tqdm import tqdm
from sklearn.metrics import cohen_kappa_score
import matplotlib.pyplot as plt
from experiments.medical_imaging.lib import datasets, models
from os import path



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def find_lr(model, optimizer, loss, dataloader, start_lr=1e-8, end_lr=10, num_iter=100):
    save = model.state_dict()
    lr_sched = lambda i: (end_lr / start_lr) ** (i / (num_iter - 1))
    schedule = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=[lr_sched])
    beta = 0.98
    best_loss = 0.
    batches_seen = 0
    losses = []
    lr = []
    smooth_losses = []
    avg_loss = 0.
    max_batches = num_iter

    for batches_seen, batch in enumerate(dataloader):
        optimizer.zero_grad()
        x, y = batch
        x, y = x.cuda(non_blocking=True), y.cuda(non_blocking=True)

        out = model(x)
        l = loss(out, y)
        l.backward()
        optimizer.step()
        losses.append(l.item())

        lr.append(optimizer.param_groups[0]["lr"])
        schedule.step()

        avg_loss = beta * avg_loss + (1 - beta) * l.item()
        smoothed_loss = avg_loss / (1 - beta ** (batches_seen + 1))
        smooth_losses.append(smoothed_loss)
        if batches_seen == 0: best_loss = smoothed_loss
        if smoothed_loss < best_loss: best_loss = smoothed_loss
        # stop when loss explodes
        if batches_seen != 0 and smoothed_loss > 4 * best_loss:
            print('loss exploded')  # mostly for debugging
            break
        if batches_seen >= max_batches - 1:
            print('last batch seen')
            break
    fig, ax = plt.subplots(1, 1)
    ax.plot(lr[5:-5], smooth_losses[5:-5])
    # ax.plot(xs[suggestion], losses[suggestion], markersize=5, marker='o', color='red')
    ax.set_xscale('log')
    ax.xaxis.set_major_formatter(plt.FormatStrFormatter('%.0e'))
    fig.show()


def train(model, dl, optim, loss):
    model.train()
    tl = []
    for x, y in tqdm(dl):
        optim.zero_grad()
        x, y = x.cuda(non_blocking=True), y.cuda(non_blocking=True)
        out = model(x)
        l = loss(out, y)
        l.backward()
        optim.step()
        tl.append(l.item())
        # schedule.step()
    return np.mean(tl)


def val(model, dl, loss):
    pred = []
    true_classes = []
    vl = []
    model.eval()
    with torch.no_grad():
        for x, y in tqdm(dl):
            x, y = x.cuda(non_blocking=True), y.cuda(non_blocking=True)
            out = model(x)
            l = loss(out, y).item()
            vl.append(l)
            pred.extend(out.cpu().numpy())
            true_classes.extend(y.cpu().numpy())

    pred = np.array(pred)
    true_classes = np.array(true_classes)
    rounded_predictions = np.argmax(pred, axis=1)
    kappa = cohen_kappa_score(true_classes, rounded_predictions, weights='quadratic')
    loss = np.mean(vl)
    return kappa, loss


if __name__ == '__main__':
    batch_size = 8
    data_loc = path.join(path.dirname(__file__), "../../data")

    ds = datasets.Aptos(img_size=320, data_location=path.join(data_loc, "APTOS"), train=True)
    class_weights = ds.class_weights
    ds_val = datasets.Aptos(img_size=320, data_location=path.join(data_loc, "APTOS"), train=False)
    dl_train = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    dl_val = torch.utils.data.DataLoader(ds_val, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    model = models.Densenet("densenet121", num_classes=5, params_loc=path.join(data_loc, "models/aptos_densenet121_weights_new.pth")) #params_loc="../data/models/aptos_densenet121_weights.pth"
    # model = models.Mobilenet_v2(True, 5)
    model.to(device)
    print(class_weights)
    loss = torch.nn.CrossEntropyLoss(weight=torch.tensor(class_weights, dtype=torch.float32,
                                                         device=device))  # weight=torch.tensor(class_weights,dtype=torch.float32,device=device)
    optimizer = optim.SGD(model.parameters(), lr=1e-13, momentum=0.9, nesterov=True)
    # optimizer = optim.Adam(model.parameters(), lr=1e-8)
    epochs = 5
    schedule = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=5e-3,
                                                   steps_per_epoch=len(dl_train), epochs=epochs, div_factor=25,
                                                   final_div_factor=30)

    best_kappa, vl = val(model,dl_val,loss)
    best_weights = model.state_dict()

    print("validation: kappa {} \t loss: {}".format(best_kappa, vl))

    # find_lr(model,optimizer,loss,dl_train,1e-13,1e-5,100)
    # exit()
    #
    # best_kappa = 0
    for e in range(epochs):
        print("epoch {}".format(e))
        tl = train(model, dl_train, optimizer, loss)
        print(" \ntrain_loss = {}".format(tl))

        kappa, vl = val(model, dl_val, loss)
        print("\nvalidation: kappa {} \t loss: {}".format(kappa, vl))
        if kappa > best_kappa:
            print("best kappa")
            best_kappa = kappa
            best_weights = model.state_dict()
    torch.save(best_weights, path.join(data_loc, "models/aptos_densenet121_weights_new.pth"))
