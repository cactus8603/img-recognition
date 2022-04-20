import os
from matplotlib.pyplot import axis
import pandas as pd
import torch
# import torch.nn as nn
import numpy as np
import random
import torch.nn.functional as F
# import timeit
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
from torchvision.transforms import Compose, Resize, ToTensor, transforms
from torch.cuda.amp import autocast as autocast
# from net import net
from PIL import Image
from tqdm import tqdm

from sklearn.metrics import precision_score
# accuracy_score, average_precision_score,f1_score,recall_score

device = "cuda" if torch.cuda.is_available() else "cpu"

def read_dict(target):
    path = './data/raw/' + str(target)
    imgname = []
    for filename in os.listdir(path):
        imgname.append(filename)

    return imgname

def getimg():
    # root = self.img_root
    dict_ = pd.read_csv('label.txt', delimiter=" ", header=None).to_dict()[0]

    imgid = []
    labelid = np.array([]).reshape(0,14)

    for classes in range(len(dict_)):
        img = read_dict(dict_[classes])
        # img = img[:5]
        label = np.zeros(len(dict_))
        label[classes] = 1
        label = np.tile(label, (len(img),1))

        imgid += img
        labelid = np.vstack((labelid, label))

    return imgid, labelid, dict_

def getdetail(idx, imgid, labelid, dict_):

    img = imgid[idx]
    label = labelid[idx]

    # dict = pd.read_csv('label.txt', delimiter=" ", header=None).to_dict()[0]
    pos = int(np.where(label==1)[0])
    path = './data/raw/' + str(dict_[pos]) + '/' + str(img)
    im = Image.open(path).copy()
    # im = im.resize((512,512))

    return im, label


def preprocess(idx):
    imgid, labelid = getimg()
    idx_random = random.choices(range(len(imgid)))[0]

    im1, label1 = getdetail(idx)
    im2, label2 = getdetail(idx_random)

    alpha = round(random.uniform(0,1), 2)
    blendimg = Image.blend(im1, im2 , alpha=alpha)
    label = label1*(1-alpha) + label2*alpha

    blendimg.show()

    return blendimg, label

def get_confusion_matrix(y_true, y_pred):
    y_true, y_pred = y_true.cpu().detach().numpy() , y_pred.cpu().detach().numpy() 
    cm = np.array(confusion_matrix(y_true, y_pred))
    tmp = np.append(y_true, y_pred)
    ele = np.unique(tmp)
    for idx in range(0,14):
        if (idx not in ele):

            cm = np.insert(cm, idx, np.zeros(cm.shape[0]), axis=1)
            cm = np.insert(cm, idx, np.zeros(cm.shape[1]), axis=0)
    # print(cm.shape)
    return cm

def WP_score(cm):
    # FP = cm.sum(axis=0) - np.diag(cm)  
    FN = cm.sum(axis=1) - np.diag(cm)
    TP = np.diag(cm)
    # TN = cm.sum() - (FP + FN + TP)

    WP = 0
    for idx in range(0,14):
        precision = float(TP[idx] / (TP[idx]+FN[idx]))
        # recall =  float(TP[idx] / (TP[idx]+FN[idx]))
        # f1 = 2 * (precision*recall) / (precision+recall)
        if precision == 'nan':
            continue
        else:
            WP += precision*(TP[idx]+FN[idx])
    
    return WP

def train_epoch(dataloader, model, loss_fn, opt, lr_scheduler, scaler):
    print("Train state")
    # size = int(len(dataloader.dataset) / 48)
    
    model.train()
    accumulation_steps = 4

    cm = np.zeros((14,14),dtype=int)

    train_transform = torch.nn.Sequential(
        # transforms.ToPILImage(),
        transforms.ColorJitter(brightness=0.35, contrast=0.35, saturation=0.2, hue=0.12),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        # transforms.RandomRotation(degrees=(-40,40), center=(250,100)),
        transforms.RandomPerspective(),
        # transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        transforms.RandomErasing(p=0.5, scale=(0.02, 0.3), ratio=(0.3, 3.3)),
    )
    # script = torch.jit.script(train_transform)

    for i, (im, label) in enumerate(tqdm(dataloader)):
        # print('-----train-----')
        im, label = im.to(device), label.to(device)
        shape = im.shape[0]
        # im = train_transform(im).unsqueeze(0)
        # print(im.size())

        try:
            im = im.view(shape,-1,512,512)
        except:
            continue

        with autocast():
            pred = model(im)
            loss = loss_fn(pred, label)
            
        
        loss /= accumulation_steps
        scaler.scale(loss).backward()

        p = F.softmax(pred, dim=1)
        cm += get_confusion_matrix(label.argmax(1), p.argmax(1))
        
        if((i+1) % accumulation_steps == 0 or (i+1 == len(dataloader))):
            scaler.step(opt)
            scaler.update()
            opt.zero_grad()

        # loss.backward()
        # opt.step()

    size = len(dataloader.dataset)
    WP = WP_score(cm) / size
    loss = loss.item()
    lr_scheduler.step(WP)
    print("Training loss:{:.5f} acc:{:.5f} lr:{}".format(loss, WP, lr_scheduler.optimizer.param_groups[0]['lr']))

def test_epoch(dataloader, model, loss_fn):
    print("Test state")

    size = len(dataloader.dataset)
    test_loss = 0
    correct = 0
    cm = np.zeros((14,14),dtype=int)
    # model.eval()

    test_transform = torch.nn.Sequential(
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    )

    with torch.no_grad():
        for im, label in tqdm(dataloader):

            im, label = im.to(device), label.to(device)
            shape = im.shape[0]
            # im = test_transform(im).unsqueeze(0)

            try:
                im = im.view(shape,-1,512,512)
            except:
                continue

            pred = model(im)
            
            with autocast():
                pred = model(im)
                loss = loss_fn(pred, label)

            p = F.softmax(pred, dim=1)
            
            # pred = torch.argmax(pred, dim=2)
            test_loss += loss.item()
            
            correct += (p.argmax(1) == label.argmax(1)).type(torch.float).sum().item()
            cm += get_confusion_matrix(label.argmax(1), p.argmax(1))
            # print(correct)
            
    test_loss /= len(dataloader)
    correct /= size
    WP = WP_score(cm) / size
    print('Test Error: Acc:{:.5f}%, WP:{:.5f}, Avg loss:{:.5f}'.format(100*correct, WP, test_loss))
    return WP
