from tkinter import Image
import torch
import torch.nn as nn
# import torch.nn.functional as F
# import matplotlib.pyplot as plt
# import numpy as np
# import random
import warnings

# from torchvision import datasets
# from PIL import Image
from torch.utils.data import DataLoader
# from torchvision.transforms import Compose, Resize, ToTensor, transforms
from torch.cuda import amp
from dataset import ImgDataset, TrainDataset, TestDataset
from function import getimg, test_epoch, train_epoch

from net import net

warnings.filterwarnings("ignore", category=FutureWarning)
if torch.cuda.is_available():
    device = torch.device('cuda')
    torch.backends.cudnn.benchmark = True
else:
    device = torch.device('cpu')

if __name__ == '__main__':
    
    torch.manual_seed(torch.initial_seed())

    imgid, labelid, dict_ = getimg('./data/train_512/')
    train_set = TrainDataset(transform=None, imgid=imgid, labelid=labelid, dict_=dict_, path='./data/train_512/', device=device)
    
    imgid, labelid, dict_ = getimg('./data/test_512/')
    test_set = TestDataset(transform=None, imgid=imgid, labelid=labelid, dict_=dict_, path='./data/test_512/', device=device)
    test1_size = int(len(test_set)*0.2)
    test2_size = len(test_set) - test1_size
    test1_set, test2_set = torch.utils.data.random_split(test_set, [test1_size, test2_size])


    batch_size = 64
    
    trainLoader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=6, pin_memory=True)
    testLoader = DataLoader(test1_set, batch_size=batch_size, shuffle=True, num_workers=6, pin_memory=True)
    # validLoader = DataLoader(valid_set, batch_size=batch_size, shuffle=True, num_workers=6, pin_memory=True)
    
    model = net().to(device)
    model.load_state_dict(torch.load('model_4_86.391.pt'))

    # 1536->512
    # 1:97.15
    # model.load_state_dict(torch.load('./model/perfect/trans/resnet34/model_1_97.00_96.88.pt'))
    # 1:97.13
    # model.load_state_dict(torch.load('./model/perfect/trans/resnet34/model_0_96.86_96.79.pt'))
    # 1:97.18
    # model.load_state_dict(torch.load('./model/perfect/trans/resnet34/model_0_96.89_96.89.pt'))
    # 1:97.17
    # model.load_state_dict(torch.load('./model/perfect/trans/resnet34/model_1_96.94_96.82.pt'))
    

    loss_fn = nn.CrossEntropyLoss(reduction='mean', label_smoothing=0.1)
    loss_fn_test = nn.CrossEntropyLoss(reduction='mean')

    opt = torch.optim.AdamW(model.parameters(), lr=1e-1)
    # lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=5, T_mult=2)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=opt, mode='min', factor=0.999, patience=5, threshold=0.001, eps=0)
    print("-----Start Training-----")

    epoch = 80
    # train
    for i in range(5,epoch):
        print("Epoch:{}".format(i))
        
        scaler = amp.GradScaler()
        train_epoch(trainLoader, model, loss_fn, opt, lr_scheduler, scaler)
        # wp = test_epoch(trainLoader, model, loss_fn_test)

        WP = test_epoch(testLoader, model, loss_fn_test)

        # print('score:{}'.format(wp*0.8+WP*0.2))
        # break
        WP = round(WP*100, 3) 
        
        modelname = 'model_' + str(i) + '_' + str(WP) + '.pt'
        torch.save(model.state_dict(), modelname)
        print('Save model {}'.format(i))
        print('\n')

