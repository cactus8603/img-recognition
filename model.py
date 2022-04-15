from turtle import forward
import torch
import torch.nn as nn
import torch.nn.functional as F
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

class STN(nn.Module):
    def __init__(self):
        super(STN, self).__init__()
        
        self.conv1 = nn.Conv2d(1,10,kernel_size=5)
        self.conv2 = nn.Conv2d(10,20,kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320,50)
        self.fc2 = nn.Linear(50,10)
        
        self.localization = nn.Sequential(
            nn.Conv2d(1,8,kernel_size=7),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(8,10,kernel_size=5),
            nn.MaxPool2d(2,stride=2),
            nn.ReLU(True)
        )
        
        self.fc_loc = nn.Sequential(
            nn.Linear(10*3*3, 32),
            nn.ReLU(True),
            nn.Linear(32, 3*2)
        )
        
        # initialize the weights/bias with identity transfromation
        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(torch.tensor([1,0,0,0,1,0], dtype=torch.float))
        
    def stn(self, x):
        xs = self.localization(x)
        xs = xs.view(-1, 10*3*3)
        theta = self.fc_loc(xs)
        theta = theta.view(-1,2,3)
        
        grid = F.affine_grid(theta, x.size())
        x = F.grid_sample(x, grid)
        
        return x
    
    def forward(self, x):
        x = self.stn(x)
        
        # perform the usual forward pass
        x = F.relu(F.max_pool2d(self.conv(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        
        return F.log_softmax(x, dim=1)
        