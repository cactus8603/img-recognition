# import pandas as pd
# import numpy as np
import torch
# import torch.nn as nn
from torch.utils.data import Dataset
from torchvision.transforms import transforms
# from PIL import Image
from function import getimg, preprocess, getdetail
# import matplotlib.pyplot as plt

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

class ImgDataset(Dataset):
    def __init__(self, transform, imgid, labelid, dict_, device, mode=None):
        super().__init__()
        # self.img_root = img_root

        # self.imgid, self.labelid = getimg()
        self.transform = transform
        self.imgid = imgid
        self.labelid = labelid
        self.dict_ = dict_
        self.mode = mode
        self.device = device

        self.train_transform = transforms.Compose([
            transforms.CenterCrop((1536,1536)),
            # transforms.RandomCrop((512,512)),
            transforms.Resize((512,512)),

            transforms.ColorJitter(brightness=0.35, contrast=0.35, saturation=0.2, hue=0.2),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            # transforms.RandomRotation(degrees=(-40,40), center=(250,100)),
            transforms.RandomPerspective(),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            transforms.RandomErasing(p=0.5, scale=(0.02, 0.3), ratio=(0.3, 3.3)),
        ])
        self.test_transform = transforms.Compose([
            transforms.CenterCrop((1536,1536)),
            transforms.Resize((512,512)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
        self.trans = transforms.Compose([
            transforms.CenterCrop((1536,1536)),
            transforms.Resize((512,512)),
            transforms.ToTensor(),
            # transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
        
        

        # self.mode = mode
    def __getitem__(self, idx):
        
        img, label = getdetail(idx, self.imgid, self.labelid, self.dict_)
        
        # img = img.to(device)
        # label = label.to(device)
        """
        if self.transform:
            img_tensor = self.transform(img).unsqueeze(0)
        else:
            trans = transforms.Compose([
                transforms.ToTensor()
            ])
            img_tensor = trans(img).unsqueeze(0)
        """
        if self.mode == 'train':
            img_tensor = self.train_transform(img).unsqueeze(0)
            # img_tensor = self.trans(img) # .unsqueeze(0)
        elif self.mode == 'test':
            img_tensor = self.test_transform(img).unsqueeze(0)
            # img_tensor = self.trans(img) # .unsqueeze(0)

        
        label_tensor = torch.from_numpy(label)

        return img_tensor, label_tensor

    def __len__(self):
        return len(self.imgid)

