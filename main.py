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
from dataset import ImgDataset
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

    imgid, labelid, dict_ = getimg('./data/pre_512/')
    imgdataset = ImgDataset(transform=None, imgid=imgid, labelid=labelid, dict_=dict_, path='./data/pre_512/', device=device)

    train_size = int(len(imgdataset)*0.8)
    test_size = len(imgdataset) - train_size
    train_set, test_set = torch.utils.data.random_split(imgdataset, [train_size, test_size])

    # score_set = imgdataset
    
    # evens = list(range(0, len(imgdataset), 10))
    # odds = list(range(1, len(imgdataset), 30))
    # train_set = torch.utils.data.Subset(imgdataset, evens)
    # test_set = torch.utils.data.Subset(imgdataset, odds)

    """valid_size = int(len(test_set)*0.5)
    test_size = len(test_set) - valid_size 
    test_set, valid_set = torch.utils.data.random_split(test_set, [valid_size, test_size])"""
    
    
    batch_size = 64
    
    trainLoader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=6, pin_memory=True)
    testLoader = DataLoader(test_set, batch_size=batch_size, shuffle=True, num_workers=6, pin_memory=True)
    # validLoader = DataLoader(valid_set, batch_size=batch_size, shuffle=True, num_workers=6, pin_memory=True)
    
    model = net().to(device)
    # model.load_state_dict(torch.load('./tmp/res34/model_55_98.81_98.947.pt'))

    # 1536->512
    # 1:97.15
    # model.load_state_dict(torch.load('./model/perfect/trans/resnet34/model_1_97.00_96.88.pt'))
    # 1:97.13
    # model.load_state_dict(torch.load('./model/perfect/trans/resnet34/model_0_96.86_96.79.pt'))
    # 1:97.18
    # model.load_state_dict(torch.load('./model/perfect/trans/resnet34/model_0_96.89_96.89.pt'))
    # 1:97.17
    # model.load_state_dict(torch.load('./model/perfect/trans/resnet34/model_1_96.94_96.82.pt'))
    


    # banana bareland carrot corn dragonfruit garlic guava peanut pineapple pumpkin rice soybean sugarcane tomato
    nSamples = [4077, 12657, 1991, 9665, 1957, 7652, 4719, 10523, 8002, 1676, 7950, 5683, 2215, 1503]
    normedWeights = [1 - (x / sum(nSamples)) for x in nSamples]
    normedWeights = torch.FloatTensor(normedWeights).to(device)

    loss_fn = nn.CrossEntropyLoss(reduction='mean', label_smoothing=0.12)
    loss_fn_test = nn.CrossEntropyLoss(reduction='mean')

    opt = torch.optim.AdamW(model.parameters(), lr=1e-1)
    # lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=5, T_mult=2)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=opt, mode='min', factor=0.995, patience=8, threshold=0.00001, eps=1e-8)
    print("-----Start Training-----")

    epoch = 80
    smooth = 0.12
    WP = 0
    # train
    for i in range(epoch):
        print("Epoch:{}".format(i))
        
        scaler = amp.GradScaler()
        train_set.dataset.mode = "train"
        if (WP > 0.8): 
            smooth *= 0.8
            if smooth < 0.01 : smooth = 0.0;
            loss_fn = nn.CrossEntropyLoss(reduction='mean', label_smoothing=smooth)
            

        train_epoch(trainLoader, model, loss_fn, opt, lr_scheduler, scaler)
        # train_set.dataset.mode = "test"
        # wp = test_epoch(trainLoader, model, loss_fn_test)

        test_set.dataset.mode = "test"
        WP = test_epoch(testLoader, model, loss_fn_test)

        # print('score:{}'.format(wp*0.8+WP*0.2))
        # break
        # WP = round(WP*100, 3) 
        if (WP > 0.85): 
            smooth *= 0.8
            loss_fn = nn.CrossEntropyLoss(reduction='mean', label_smoothing=smooth)

        

        modelname = './tmp/se/model_' + str(i) + '_' + str(WP) + '.pt'
        torch.save(model.state_dict(), modelname)
        print('Save model {}'.format(i))
        print('\n')

