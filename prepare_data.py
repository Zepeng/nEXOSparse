# prepare-data script modified for nEXO signal/background classification
# Copyright 2016-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import glob, math, os, random
import pickle
import argparse
#import pandas as pd
from PIL import Image
from scipy import sparse

batch_size = 1000

def split_files():
    imagelist = glob.glob('./sparse_img/*jpg')
    if not os.path.exists('./sparse_img/train.txt'):
        trainlist = open('./sparse_img/train.txt', 'w')
        testlist = open('./sparse_img/test.txt', 'w')
        for img_file in imagelist:
            if random.random() > 0.1:
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
    for index, line in enumerate(f):
        singleimg = {}
        if index <= batch_num*batch_size or index > (1+batch_num)*(batch_size):
            continue
        # Get image name from the pandas df
        single_image_name = line.split()[0]
        # Open image
        img_as_img = Image.open(single_image_name)
	    # data augmentation
        npimg = np.array(img_as_img)
        coords=[]
        col=[]
        cl=[]
        sx = sparse.csc_matrix(npimg[:,:,0])
        sy = sparse.csc_matrix(npimg[:,:,1])
        coords_x = np.concatenate((sx.nonzero()[0] + 450, sy.nonzero()[0]))
        coords_y = np.concatenate((sx.nonzero()[1], sy.nonzero()[1]))
        col = np.concatenate((sx.data, sy.data))
        coords = sparse.csc_matrix((col, (coords_x, coords_y)))
        if 'signal' in single_image_name:
            cl.append(1)
        else:
            cl.append(0)
        _item = (coords, cl[0])
        pickle.dump(_item, open ("./data_new/img%s_%d.p" % (img_type, index), "wb"))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Prepare data.')
    parser.add_argument('--bn', type=int, default=0, help='batch number')
    parser.add_argument('--img_type', type=str, default='train', help='batch number')
    args = parser.parse_args()
    pickle_img('./sparse_img/%s.txt' % args.img_type, args.bn, args.img_type)
