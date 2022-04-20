import torch
import torch.nn as nn
import torchvision.models as models

class net(nn.Module):
    def __init__(self):
        super(net, self).__init__()
        # self.model = models.resnet18(pretrained=False)
        self.model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet34', pretrained=False)
        # self.conv1 = nn.Conv2d()
        self.model.fc = nn.Linear(512,14)
        
    def forward(self, x):
        x = self.model(x)
        return x