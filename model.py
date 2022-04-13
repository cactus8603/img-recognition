from turtle import forward
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
"""
class imgDataset(Dataset):
    def __init__(self) -> None:
        self.data = 

    def __getitem__(self, idx):

    def __len__(self):
        return len(self.data)
        """

class net(nn.Module):
    def __init__(self):
        super(net, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5, stride=1, padding=0)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=2)

        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, stride=1, padding=0)
        
        self.fc = nn.Linear(in_features=32*4*4, out_features=10)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x