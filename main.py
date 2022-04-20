import torch
import torch.nn as nn
# import torch.nn.functional as F
# import matplotlib.pyplot as plt
# import numpy as np
# import random

# from torchvision import datasets
# from PIL import Image
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Resize, ToTensor, transforms
from torch.cuda import amp
from dataset import ImgDataset
from function import getimg, test_epoch, train_epoch

from net import net

if torch.cuda.is_available():
    device = torch.device('cuda')
    torch.backends.cudnn.benchmark = True
else:
    device = torch.device('cpu')

if __name__ == '__main__':
    torch.manual_seed(torch.initial_seed())

    imgid, labelid, dict_ = getimg()
    imgdataset = ImgDataset(transform=None, imgid=imgid, labelid=labelid, dict_=dict_, device=device)

    train_size = int(len(imgdataset)*0.8)
    test_size = len(imgdataset) - train_size
    train_set, test_set = torch.utils.data.random_split(imgdataset, [train_size, test_size])

    
    """evens = list(range(0, len(imgdataset), 20))
    odds = list(range(1, len(imgdataset), 60))
    train_set = torch.utils.data.Subset(imgdataset, evens)
    test_set = torch.utils.data.Subset(imgdataset, odds)"""
    

    """valid_size = int(len(test_set)*0.5)
    test_size = len(test_set) - valid_size 
    test_set, valid_set = torch.utils.data.random_split(test_set, [valid_size, test_size])"""
    
    
    batch_size = 64
    
    trainLoader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=6, pin_memory=True)
    testLoader = DataLoader(test_set, batch_size=batch_size, shuffle=True, num_workers=6, pin_memory=True)
    # validLoader = DataLoader(valid_set, batch_size=batch_size, shuffle=True, num_workers=6, pin_memory=True)
    
    model = net().to(device)
    # model.load_state_dict(torch.load('./model/model_0.pt'))
    model.load_state_dict(torch.load('model_0_96.pt'))
    loss_fn = nn.CrossEntropyLoss(reduction='mean', label_smoothing=0.1)
    loss_fn_test = nn.CrossEntropyLoss(reduction='mean')

    opt = torch.optim.AdamW(model.parameters(), lr=0.001)
    # lr_scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=3, gamma=0.5)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=opt, mode='max', factor=0.1, patience=3, threshold=0.01)
    print("-----Start Training-----")
    epoch = 99
    # train
    for i in range(epoch):
        print("Epoch:{}".format(i))
        
        scaler = amp.GradScaler()
        train_set.dataset.mode = "train"
        train_epoch(trainLoader, model, loss_fn, opt, lr_scheduler, scaler)

        test_set.dataset.mode = "test"
        WP = test_epoch(testLoader, model, loss_fn_test)
        WP = round(WP*100, 3) 

        # 24: 86.89%
        # 25: 86.47%
        # 27: 86.92%

        modelname = 'model_' + str(i) + '_' + str(WP) + '.pt'
        torch.save(model.state_dict(), modelname)
        print('Save model {}'.format(i))
        print('\n')

