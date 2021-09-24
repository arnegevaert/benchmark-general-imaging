import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import torchvision.models
from experiments.medical_imaging.lib.models import EfficientNet, senet154
from experiments.medical_imaging.lib.datasets import HAM10000
import os
import numpy as np
import argparse
from collections import deque
import time
import sklearn.metrics as metrics
import copy

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

#### settings used for efficientnet-b0: batchsize 40, lr 5e-5

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model',required=True)
    parser.add_argument('--param_loc',type=str)
    parser.add_argument('--data_loc', required=True, type=str)
    parser.add_argument('-b','--batch_size', default=32, type=int)
    parser.add_argument('--epochs', default=20, type=int)
    parser.add_argument('--lr', default=1e-3, type=float)
    parser.add_argument('--validate', action="store_true")
    return parser

def train_epoch(net, opt,crit,dl):
    # scaler = torch.cuda.amp.GradScaler(enabled=True)
    net.train()
    losses=[]
    for batch,labels in dl:
        batch, labels = batch.to(device), labels.type(torch.long).to(device)
        opt.zero_grad()
        out = net(batch)
        loss = crit(out,labels)
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
    optimizer = optim.AdamW(model.parameters(), lr=args.lr) #weight_decay=1e-4
    # optimizer = optim.SGD(model.parameters(),lr=args.lr,momentum=0.9,nesterov=True)

    #select best model based on average sensitivity
    # schedule = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=1, threshold=0.001, factor=0.2, verbose=True,mode='max')
    schedule = optim.lr_scheduler.StepLR(optimizer,step_size=25,gamma=0.5,verbose=True)

    best_weights, best_Sens = None, -float("inf")
    counter = 0
    for e in range(args.epochs):
        tic()
        print("Epoch {}/{}".format(e, args.epochs))
        print("-" * 10)
        train_loss = train_epoch(model, optimizer, criterion, train_dl)
        Sens,bal_acc,val_loss = calculate_metrics(model, criterion, val_dl)
        toc()
        schedule.step()

        if Sens > best_Sens:
            best_weights = copy.deepcopy(model.state_dict())
            best_Sens = Sens
            best_e = e
            counter = 0
            print("Model improved")
        else:
            counter += 1
        if counter > 40:
            print("No improvements: stopping")
            break
    torch.save(best_weights, args.model + '_epoch_{}.pt'.format(best_e))
    print("Best Validation Sens:", best_Sens)
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
    AUC = metrics.roc_auc_score(gt,preds,average='macro', multi_class='ovo')
    print('avg_Sens: {:.4f} AUC: {:.4f} Loss: {:.4f}'.format(Sens,AUC,val_loss))
    return Sens,AUC,val_loss

def run(args):


    if args.model == 'senet154':
        pass
        model = senet154(7, params_loc=args.param_loc)
    elif "efficientnet" in args.model:
        model = EfficientNet(args.model,7,params_loc=args.param_loc)
    else:
        raise Exception("{} is not a valid model.".format(args.model))
    model.to(device)

    train_ds = HAM10000(args.data_loc,train=True,imsize=224)
    train_dl = DataLoader(train_ds,batch_size=args.batch_size,shuffle=True,num_workers=2,pin_memory=True)
    val_ds = HAM10000(args.data_loc, train=False, imsize=224)
    val_dl = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)

    class_weights = torch.FloatTensor(train_ds.class_weights).to(device)
    criterion = torch.nn.CrossEntropyLoss(class_weights)
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