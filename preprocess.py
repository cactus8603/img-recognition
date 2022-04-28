
from ctypes import Union
import pandas as pd
import os
from os.path import exists
from tqdm import tqdm, trange
import time
import shutil

from PIL import Image
from torchvision.transforms import Compose, Resize, ToTensor, transforms

def resample(idx):
    
    trans = transforms.Compose([
            # transforms.CenterCrop((1536,1536)),
            # transforms.RandomCrop((512,512)),
            transforms.Resize((512,512)),
        ])
    dict_ = pd.read_csv('label.txt', delimiter=" ", header=None).to_dict()[0]
    
    nSamples = [4046, 12482, 1890, 9665, 1957, 7652, 4719, 10523, 8002, 1676, 7950, 5683, 2215, 1503]
    # total = sum(nSamples)
    progress = tqdm(total=nSamples[idx])
    # for label in range(14):
    load_path = './data/raw/' + str(dict_[idx])

    for filename in os.listdir(load_path):
    
        progress.update(1)

        load_path = './data/raw/' + str(dict_[idx]) + '/' + str(filename)
        im = Image.open(load_path).convert('RGB').copy()
        im = trans(im)

        save_path = './data/pre_512/' + str(dict_[idx]) + '/' + str(filename)
        # if(os.path.exists(save_path)): save_path = './data/pre_512/' + str(dict_[idx]) + '/1-' + str(filename)
        im = im.save(save_path, subsampling=0)
        # im.show()

            # break
        # imgname.append(filename)

def deleteimg(idx):
    dict_ = pd.read_csv('label.txt', delimiter=" ", header=None).to_dict()[0]

    com1 = pd.read_csv('output/97009688/'+dict_[idx]+'.csv', delimiter=",", header=None)
    com1 = com1[0].to_list()
    com2 = pd.read_csv('output/96899689/'+dict_[idx]+'.csv', delimiter=",", header=None)
    com2 = com2[0].to_list()
    deletefile = sorted(list(set(com1) & set(com2)))
    # print(deletefile)
    # print(len(deletefile))

    progress = tqdm(total=len(deletefile))

    srcpath = './data/pre_1536/' + str(dict_[idx]) + '/'
    despath = './data/pre_1536/others/' + str(dict_[idx]) + '/'
    for filename in deletefile:
        progress.update(1)
        progress.set_postfix({'類別':str(dict_[idx])})
        
        src = srcpath + filename
        des = despath + filename

        if os.path.exists(src):
            shutil.move(src, des)


def delete(idx):
    dict_ = pd.read_csv('label.txt', delimiter=" ", header=None).to_dict()[0]

    com1 = pd.read_csv('1.csv', delimiter=",", header=None)
    com1 = set(com1[0].to_list())
    com2 = pd.read_csv('2.csv', delimiter=",", header=None)
    com2 = set(com2[0].to_list())
    deletefile = sorted(list(com1.union(com2)))
    # print(deletefile)
    # print(len(deletefile))

    progress = tqdm(total=len(deletefile))

    srcpath = './data/train_512/' + str(dict_[idx]) + '/'
    
    for filename in deletefile:
        progress.update(1)
        progress.set_postfix({'類別':str(dict_[idx])})
        
        src = srcpath + filename

        if os.path.exists(src):
            os.remove(src)
    


if __name__ == '__main__':

    dict_ = pd.read_csv('label.txt', delimiter=" ", header=None).to_dict()[0]

    # for idx in range(14):
    #     deleteimg(idx)

    # add folder
    # for label in range(14):
    #     path = './data/pre_512/' + str(dict_[label])
    #     if not os.path.isdir(path):
    #         os.makedirs(path)

    # label = 0
    # idx = [2,4,5,6,7,8,10,11,12,13]
    for i in range(14):
        resample(i)

    # delete(13)
    # resample(12)
    # banana bareland carrot corn dragonfruit garlic guava peanut pineapple pumpkin rice soybean sugarcane tomato
    # nSamples = [4077, 12657, 1991, 9665, 1957, 7652, 4719, 10523, 8002, 1676, 7950, 5683, 2215, 1503]
                                                        #
    # resample(6)
    
    
