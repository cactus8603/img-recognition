import torch
import torch.nn as nn
import torchvision.models as models
import timm
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

class net(nn.Module):
    def __init__(self):
        super(net, self).__init__()
        # self.model = models.resnet18(pretrained=False)
        self.model = timm.create_model('seresnext26d_32x4d', pretrained=False, num_classes=14)
        # self.model = timm.create_model('res2next50', pretrained=False, num_classes=14)
        # self.model = torch.hub.load('pytorch/vision:v0.10.0', 'resnext50_32x4d', pretrained=False) # resnext
        self.model.fc = nn.Linear(2048,14)

        # self.model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet34', pretrained=False)
        # self.model.fc = nn.Linear(512,14)
        
        
    def forward(self, x):
        x = self.model(x)
        return x