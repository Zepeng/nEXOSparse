# prepare-data script modified for nEXO signal/background classification
# Copyright 2016-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import glob, os

def split_files():
    imagelist = glob.glob('./data/train_*p')
    trainlist = open('train.txt', 'w')
    i = 0
    for img_file in imagelist:
        i+=1
        trainlist.write(img_file)
        trainlist.write('\n')
        if i > 30000:
            break
split_files()
