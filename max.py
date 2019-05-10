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
from PIL import Image

batch_size = 1000

def split_files():
    maxpixel = 0
    imagelist = glob.glob('./sparse_img/*jpg')
    for single_image_name in imagelist:
        img_as_img = Image.open(single_image_name)
	    # data augmentation
        npimg = np.array(img_as_img)
        if np.amax(npimg) > maxpixel:
            maxpixel = np.amax(npimg)
        print(maxpixel)
if __name__ == '__main__':
    split_files()
