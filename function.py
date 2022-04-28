import os
# from matplotlib.pyplot import axis
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
import matplotlib.pyplot as plt
import warnings

from sklearn.metrics import precision_score

warnings.filterwarnings("ignore", category=FutureWarning)
device = "cuda" if torch.cuda.is_available() else "cpu"

def read_dict(target, path):
    path += str(target)
    # path = './data/preprocessed_512/' + str(target)
    # path = './data/pre_1536/' + str(target)

    imgname = []
    for filename in os.listdir(path):
        imgname.append(filename)

    return imgname

def getimg(path):
    # root = self.img_root
    dict_ = pd.read_csv('label.txt', delimiter=" ", header=None).to_dict()[0]

    imgid = []
    labelid = np.array([]).reshape(0,14)

    for classes in range(len(dict_)):
        img = read_dict(dict_[classes], path)
        # img = img[:5]
        label = np.zeros(len(dict_))
        label[classes] = 1
        label = np.tile(label, (len(img),1))

        imgid += img
        labelid = np.vstack((labelid, label))

    return imgid, labelid, dict_

def getdetail(idx, imgid, labelid, dict_, path):

    img = imgid[idx]
    label = labelid[idx]

    pos = int(np.where(label==1)[0])
    path += str(dict_[pos]) + '/' + str(img)
    # path = './data/preprocessed_512/' + str(dict_[pos]) + '/' + str(img)
    # path = './data/pre_1536/' + str(dict_[pos]) + '/' + str(img)

    im = Image.open(path).convert('RGB').copy()
    # im = im.resize((512,512))

    return im, label


def get_confusion_matrix(y_true, y_pred):
    y_true, y_pred = y_true.cpu().detach().numpy() , y_pred.cpu().detach().numpy() 
    cm = np.array(confusion_matrix(y_true, y_pred),dtype=float)
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
        # recall =  float(TP[idx] / (TP[idx]+FP[idx]))
        # f1 = 2 * (precision*recall) / (precision+recall)
        # print("Type:{}, f1-score:{}".format(idx+1, f1))

        WP += precision*(TP[idx]+FN[idx])
    
    return WP

def train_epoch(dataloader, model, loss_fn, opt, lr_scheduler, scaler):
    print("Train state")
    # size = int(len(dataloader.dataset) / 2)
    
    model.train()
    accumulation_steps = 4

    # cm = np.zeros((14,14),dtype=float)
    correct = 0
    train_loss = 0
    avg_loss = 0
    count = tqdm(dataloader)
    for i, (im, label) in enumerate(count):
        # im = trans(im).unsqueeze(0)
        # print('-----train-----')
        im, label = im.to(device), label.to(device)
        shape = im.shape[0]
        img_size = im.shape[3]

        
        try:
            im = im.view(shape,-1,img_size,img_size)
        except:
            pass

        with autocast():
            pred = model(im)
            loss = loss_fn(pred, label)

        train_loss += loss.item()
        loss /= accumulation_steps
        scaler.scale(loss).backward()
        

        p = F.softmax(pred, dim=1)
        correct += (p.argmax(1) == label.argmax(1)).type(torch.float).sum().item()
        # cm += get_confusion_matrix(label.argmax(1), p.argmax(1))
        # WP = WP_score(cm) / (i*64 + shape)

        if((i+1) % accumulation_steps == 0 or (i+1 == len(dataloader))):
  
            scaler.step(opt)
            scaler.update()
            opt.zero_grad()
            avg_loss = float(train_loss / (i+1))
            lr_scheduler.step(loss.item())
            # lr_scheduler.step(WP)
            count.set_postfix({'lr':round(lr_scheduler.optimizer.param_groups[0]['lr'],10)})
            count.set_description('loss:{:.5f}'.format(avg_loss))
            # count.set_description('acc:{:.5f}'.format(WP))
            # print('lr:', lr_scheduler.optimizer.param_groups[0]['lr'])
            # print('loss:', loss.item())
        

        # loss.backward()
        # opt.step()
    # lr_scheduler.step()
    size = len(dataloader.dataset)
    # WP = WP_score(cm) / size
    # print(cm)
    
    print("Training loss:{:.5f} acc:{:.5f}% lr:{:.5f}".format(avg_loss, correct/size, lr_scheduler.optimizer.param_groups[0]['lr']))

def test_epoch(dataloader, model, loss_fn):
    print("Test state")

    size = len(dataloader.dataset)
    test_loss = 0
    correct = 0
    cm = np.zeros((14,14),dtype=float)
    model.eval()

    with torch.no_grad():
        for im, label in tqdm(dataloader):

            im, label = im.to(device), label.to(device)
            shape = im.shape[0]
            img_size = im.shape[3]
            
            try:
                im = im.view(shape,-1,img_size,img_size)
            except:
                pass

            
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
    # print(cm)
    # for i in range(14):
        # print(i, cm[i][i])
    print('Test Error: Acc:{:.5f}%, WP:{:.5f}, Avg loss:{:.5f}'.format(100*correct, WP, test_loss))
    return WP

