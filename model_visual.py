from torchsummary import summary
from net import net
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = net().to(device)
summary(model, input_size=(3,512,512))