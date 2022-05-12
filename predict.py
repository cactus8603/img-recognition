import os
import torch.nn.functional as F
from PIL import Image
import torch
import numpy as np
import pandas as pd
from torchvision.transforms import transforms
from net import net
from tqdm import tqdm

if torch.cuda.is_available():
    device = torch.device('cuda')
    torch.backends.cudnn.benchmark = True
else:
    device = torch.device('cpu')


print("-----Start Predicting-----")




test_transform = transforms.Compose([
            # transforms.CenterCrop((1536,1536)),
            # transforms.Resize((512,512)),
            transforms.ToTensor(),
            transforms.Normalize([0.439, 0.459, 0.406], [0.185, 0.186, 0.229])
        ])
dict_ = pd.read_csv('label.txt', delimiter=" ", header=None).to_dict()[0]

model = net().to(device)
# model.load_state_dict(torch.load('./model/perfect/trans/resnet34/model_1_97.00_96.88.pt'))
model.load_state_dict(torch.load('./model/resnet34/res34/model_55_98.81_98.947.pt'))
model.eval()

nSamples = [4077, 12657, 1991, 9665, 1957, 7652, 4719, 10523, 8002, 1676, 7950, 5683, 2215, 1503]


for idx in range(14):
    idx = 0
    imgname = []
    label = []
    progress = tqdm(total=nSamples[idx])
    path = './data/pre_512/' + str(dict_[idx])
    progress.set_postfix({'類別':str(dict_[idx])})

    no = []
    co = []
    for filename in os.listdir(path):
        progress.update(1)
        imgpath = path + '/' + filename
        # print(imgpath)
        
        with torch.no_grad():
            im = Image.open(imgpath).copy()
            im = test_transform(im).unsqueeze(0).to(device)
            pred = model(im)
            p = F.softmax(pred, dim=1)
            pred = dict_[int(p.argmax(1))]
            # print(p)
            # print(filename)
            # print(pred)
            # if (pred != dict_[idx]):
            # #     print(p)
            # #     print(p[0][int(p.argmax(1))])
            # #     print(filename)
            # #     print(pred)
            #     if(p[0][int(p.argmax(1))]*0.918 < 0.3):
            #         no.append(round(float(p[0][int(p.argmax(1))]),3))
            #     # continue
            # elif (pred == dict_[idx]):
            #     if(p[0][int(p.argmax(1))]*0.918 < 0.3):
            #         co.append(round(float(p[0][int(p.argmax(1))]),3))

        imgname.append(filename)
        label.append(pred)
        
    # im.show()
    # break
    d = {'filename':imgname, 'pred':label}
    result = pd.DataFrame(data=d)
    filename = './' + str(dict_[idx]) + '.csv'
    result.to_csv(filename, index=False, header=False)
    # print(no)
    # print(len(no))
    # print(co)
    # print(len(co))
    break

    