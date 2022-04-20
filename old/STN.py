from __future__ import print_function
from msilib import datasizemask
from turtle import down
from black import main
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
import numpy as np
from model import STN

plt.ion()

from six.moves import urllib


def train(epoch, model):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        
        opt.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        opt.step()
        if batch_idx % 500 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)] \t Loss:{:.6f}'.format(epoch, batch_idx*len(data), len(train_loader.dataset), 100.*batch_idx / len(train_loader), loss.item()))
            
def test(model):
    with torch.no_grad():
        model.eval()
        test_loss = 0
        correct = 0
        
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            
            # sum up the batch loss
            test_loss += F.nll_loss(output, target, size_average=False).item()
            
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()
            
        test_loss /= len(test_loader.dataset)
        print('\nTest set: Average loss:{:.4f}, Accuracy:{}/{}({:.0f}%)\n'.format(test_loss, correct, len(test_loader.dataset), 100. * correct / len(test_loader.dataset)))
        
def convert_image_np(inp):
    # convert to numpy image
    inp = inp.numpy().transpose((1,2,0))
    mean = np.array([0.485,0.456,0.406])
    std = np.array([0.229,0.224,0.225])
    inp = std*inp + mean
    inp = np.clip(inp, 0, 1)
    return inp

def visualize_stn():
    with torch.no_grad():
        data = next(iter(test_loader))[0].to(device)
        
        input_tensor = data.cpu()
        transformed_input_tensor = model.stn(data).cpu()
        
        in_grid = convert_image_np(
            torchvision.utils.make_grid(input_tensor)
        )
        
        out_grid = convert_image_np(
            torchvision.utils.make_grid(transformed_input_tensor)
        )
        
        f, axarr = plt.subplot(2)
        axarr[0].imshow(in_grid)
        axarr[0].set_title('Dataset Images')
        
        axarr[1].imshow(out_grid)
        axarr[1].set_title('Transformed Images')
    plt.ioff()
    plt.show()

if __name__ == '__main__':
    opener = urllib.request.build_opener()
    opener.addheaders = [('User-agent', 'Mozilla/5.0')]
    urllib.request.install_opener(opener)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Training dataset
    train_loader = DataLoader(
        datasets.MNIST(root='.', train=True, download=True, 
                    transform = transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize((0.1307,),(0.3081,))
                    ])), batch_size=64, shuffle=True, num_workers=4
    )

    test_loader = DataLoader(
        datasets.MNIST(root='.', train=False, 
                    transform = transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize((0.1307,),(0.3081,))
                    ])), batch_size=64, shuffle=True, num_workers=4
    )

    model = STN().to(device)
    opt = optim.AdamW(model.parameters(), lr=0.01)
    
    for epoch in range(1,3+1):
        train(epoch, model)
        test(model)
        
    visualize_stn()
    