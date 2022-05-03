from matplotlib.pyplot import axis
from tqdm import tqdm
import os
import pandas as pd
from PIL import Image, ImageStat
import numpy as np

nSamples = [4077, 12657, 1991, 9665, 1957, 7652, 4719, 10523, 8002, 1676, 7950, 5683, 2215, 1503]
dict_ = pd.read_csv('label.txt', delimiter=" ", header=None).to_dict()[0]

mean = np.zeros((1,3))
std = np.zeros((1,3))
out_mean = np.zeros((1,3))
out_std = np.zeros((1,3))
for idx in range(14):
    # idx = 13
    # imgname = []
    # label = []
    progress = tqdm(total=nSamples[idx])
    path = './data/pre_512/' + str(dict_[idx])
    progress.set_postfix({'類別':str(dict_[idx])})

    for filename in os.listdir(path):
        progress.update(1)
        imgpath = path + '/' + filename
        im = Image.open(imgpath).copy()   
        stat = ImageStat.Stat(im)
        mean += stat.mean
        std += stat.stddev

        
    out_mean += (mean/255)
    out_std += (std/255)

    # print((mean/nSamples[idx])/255)
    # print((std/nSamples[idx])/255)

print(out_mean/sum(nSamples))
print(out_std/sum(nSamples))