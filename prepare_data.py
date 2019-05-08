# prepare-data script modified for nEXO signal/background classification
# Copyright 2016-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import torch
import glob, math, os, random
import pickle
import argparse
#import pandas as pd
import numpy as np
from PIL import Image

batch_size = 1000

def split_files():
    imagelist = glob.glob('./sparse_img/*jpg')
    if not os.path.exists('./sparse_img/train.txt'):
        trainlist = open('./sparse_img/train.txt', 'w')
        testlist = open('./sparse_img/test.txt', 'w')
        for img_file in imagelist:
            if random.random() > 0.2:
                trainlist.write(img_file)
                if 'signal' in img_file:
                    trainlist.write(' 1\n')
                else:
                    trainlist.write(' 0\n')
            else:
                testlist.write(img_file)
                if 'signal' in img_file:
                    testlist.write(' 1\n')
                else:
                    testlist.write(' 0\n')

split_files()

def pickle_img(img_list, batch_num, img_type):
    f = open(img_list,'r')
    tosave = []
    for index, line in enumerate(f):
        singleimg = {}
        if index <= batch_num*batch_size or index > (1+batch_num)*(batch_size):
            continue
        print(line)
        # Get image name from the pandas df
        single_image_name = line.split()[0]
        # Open image
        img_as_img = Image.open(single_image_name)
	    # data augmentation
        npimg = np.array(img_as_img)
        coords=[]
        col=[]
        cl=[]
        _item = {}
        for x in range(500):
            for y in range(1700):
                for z in range(2):
                    if npimg[x, y, z] > 0:
                        if z == 1:
                            coords.append([x,y+500])
                        else:
                            coords.append([x, y])
                        col.append(npimg[x, y, z])
        coords=np.array(coords,dtype='int16')
        col=np.array(col,dtype='uint8')
        if 'signal' in single_image_name:
            cl.append(1)
        else:
            cl.append(0)
        singleimg['coords'] = coords
        singleimg['features'] = col
        pickle.dump(singleimg, open ("./data/img%s_%d.p" % (img_type, index), "wb"))
        _item['img'] = "./data/img%s_%d.p" % (img_type, index)
        _item['target'] = cl[0]
        tosave.append(_item)
        print(len(tosave))
    pickle.dump( tosave, open( "./data/%s_%d.p" % (img_type, batch_num), "wb" ) )

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Prepare data.')
    parser.add_argument('--bn', type=int, default=0, help='batch number')
    parser.add_argument('--img_type', type=str, default='train', help='batch number')
    args = parser.parse_args()
    pickle_img('./sparse_img/train.txt', args.bn, args.img_type)
