import torch
from torch.utils.data import DataLoader
import torch.optim as optim
from torchvision import transforms, datasets
from experiments.general_imaging.models import Resnet20
from experiments.general_imaging.dataset_models import Resnet18
import os
import numpy as np
import argparse
from collections import deque
import time
import sklearn.metrics as metrics
import copy
import imgaug.augmenters as iaa

device = "cuda"
start_time = deque()
def tic():
    start_time.append(time.time())
def toc():
    try:
        st = start_time.pop()
        t = time.time() - st
        print("elapsed time: {}".format(time.strftime("%H:%M:%S",time.gmtime(t))))
        return t
    except:
        return None


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model',required=True)
    parser.add_argument('--param_loc',type=str)
    parser.add_argument('--data_loc', required=True, type=str)
    parser.add_argument('-b','--batch_size', default=32, type=int)
    parser.add_argument('--epochs', default=20, type=int)
    parser.add_argument('--lr', default=1e-3, type=float)
    parser.add_argument('--validate', action="store_true")
    parser.add_argument('--dropout', action="store_true")
    return parser

def train_epoch(net, opt,crit,dl, dropout):
    net.train()
    losses=[]
    for batch,labels in dl:
        labels = labels.type(torch.long).to(device)
        batch = batch.to(device)

        out = net(batch)
        loss = crit(out,labels)
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
    train_loss = np.mean(losses)
    print("Training Loss: {:.6f}".format(train_loss))
    return train_loss

def validate_epoch(net,crit,dl):
    losses = []
    net.eval()
    with torch.no_grad():
        for batch, labels in dl:
            batch, labels = batch.to(device), labels.type(torch.long).to(device)
            out = net(batch)
            loss = crit(out, labels)
            losses.append(loss.item())
    val_loss = np.mean(losses)
    print("Validation Loss: {:.6f}".format(val_loss))
    return val_loss

def train_loop(args, criterion, model, train_dl, val_dl):
    # optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    optimizer = optim.SGD(model.parameters(), lr = args.lr,momentum=0.9, weight_decay= 1e-4)
    schedule = optim.lr_scheduler.MultiStepLR(optimizer,milestones=[80,120],gamma=0.1)
    # schedule = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, threshold=0.001, factor=0.1, verbose=True)

    best_weights, best_loss = None, float("inf")
    counter = 0
    for e in range(args.epochs):
        tic()
        print("Epoch {}/{}".format(e, args.epochs))
        print("-" * 10)
        train_loss = train_epoch(model, optimizer, criterion, train_dl, args.dropout)
        Sens,bal_acc,val_loss = calculate_metrics(model, criterion, val_dl)
        toc()
        schedule.step(val_loss)

        if val_loss < best_loss:
            best_weights = copy.deepcopy(model.state_dict())
            best_loss = val_loss
            best_e = e
            counter = 0
            print("Model improved")
        # else:
        #     counter += 1
        # if counter > 22:
        #     print("No improvements: stopping")
        #     break
    torch.save(best_weights, args.model + '_epoch_{}_val{}.pt'.format(best_e,best_loss))
    print("Best Validation Sens:", best_loss)
    print("at epoch:", best_e)
    model.load_state_dict(best_weights)

def calculate_metrics(model,crit,dl):
    gt=[]
    preds=[]
    losses = []
    model.eval()
    with torch.no_grad():
        for batch, labels in dl:
            batch, labels = batch.to(device), labels.type(torch.long).to(device)
            out = model(batch)
            loss = crit(out, labels)
            losses.append(loss.item())
            out = torch.softmax(out,1).detach().cpu().numpy()
            preds.append(out)
            gt.extend(labels.detach().cpu().numpy())
    preds = np.vstack(preds)
    gt = np.array(gt)
    val_loss = np.mean(losses)
    Sens = metrics.recall_score(gt,preds.argmax(axis=1),average='macro')
    acc = metrics.accuracy_score(gt,preds.argmax(axis=1))
    print('avg_Sens: {:.4f} acc: {:.4f} Loss: {:.4f}'.format(Sens,acc,val_loss))
    return Sens,acc,val_loss

def run(args):


    if args.model == 'resnet18':
        model = Resnet18(10,params_loc=args.param_loc)
    elif args.model == 'resnet20':
        model = Resnet20(10,params_loc=args.param_loc)
    else:
        raise Exception("{} is not a valid model.".format(args.model))
    model.to(device)
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4821, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    train_ds= datasets.CIFAR10(args.data_loc,train=True, transform=transform_train,download=False)

    if args.dropout:
        train_ds = DropoutDataset(train_ds)
    train_dl = DataLoader(train_ds,batch_size=args.batch_size,shuffle=True,num_workers=2,pin_memory=True)
    transform_val = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4821, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    val_ds = datasets.CIFAR10(args.data_loc,train=False, transform=transform_val,download=False)
    val_dl = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=1, pin_memory=True)


    criterion = torch.nn.CrossEntropyLoss()
    # criterion = torch.nn.MultiMarginLoss()
    if not args.validate:
        # train_top(args, criterion, model, train_dl, val_dl)
        train_loop(args, criterion, model, train_dl, val_dl)

    calculate_metrics(model,criterion,val_dl)

if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    assert os.path.exists(args.data_loc)
    run(args)