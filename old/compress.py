
from PIL import Image
from glob import glob
from os.path import splitext
import os


def compress(path1, path2):
    jpglist = glob("data/or/*.[jJ][pP][gG]")
    for jpg in jpglist:
        im = Image.open(jpg)
        pic = 'data/com/' + splitext(jpg)[0][8:] + "." + 'jpg'

        width = 1000
        ratio = float(width)/im.size[0]
        height = int(im.size[1]*ratio)
        nim = im.resize( (width, height), Image.BILINEAR )
        print (nim.size)
        nim.save(pic,quality=100)
        # im.save(pic)
        # print(pic)



if __name__ == '__main__':
    compress('data/or/', 'data/com/')
    # img = Image.open('data/or/160105-1-0103.JPG')
    # img.save('data/com/160105-1-0103.JPG',quality=65)
    