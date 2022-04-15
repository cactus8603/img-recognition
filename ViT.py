import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from torch import nn, Tensor
from PIL import Image
from torchvision.transforms import Compose, Resize, ToTensor
from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange, Reduce
from function import *
from torchsummary import summary

img = Image.open('./data/banana/160107-3-0035.JPG', 'r')
width, height = img.size
print(width, height)
plt.imshow(img)
# plt.show()

trans = Compose([Resize((224,224)), ToTensor()])
x = trans(img)
x = x.unsqueeze(0)
print(x.shape)

# patch_size = 16
# pathes = rearrange(x, 'b c (h s1) (w s2) -> b (h w) (s1 s2 c)', s1=patch_size, s2=patch_size)
# print(pathes)
# 

summary(ViT(), (3,224,224))