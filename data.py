# Copyright 2016-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch, torch.utils.data
import sparseconvnet as scn
import pickle
import math
import random
import numpy as np
import os

class Data(torch.utils.data.Dataset):
    def __init__(self,file):
        self.imglist = open(file).readlines()
        self.num_img = len(self.imglist)
    def __getitem__(self,index):
        imgfile = self.imglist[index].strip()
        p = pickle.load(open(imgfile, 'rb'))
        item = {}
        item['target'] = p[1]
        msparse = p[0]
        item['coords'] = torch.cat([torch.from_numpy(msparse.nonzero()[0]).view(-1,1), torch.from_numpy(msparse.nonzero()[1]).view(-1,1)], 1)*1/3.
        features = torch.from_numpy(msparse.data).view(-1,1)
        item['features'] = torch.cat([features, features], 1)
        return item
    def __len__(self):
        return self.num_img

def MergeFn():
    def merge(tbl):
        v=torch.Tensor([[1,0,0]])
        targets=[x['target'] for x in tbl]
        locations=[]
        features=[]
        for idx,char in enumerate(tbl):
            m = torch.eye(2)
            coords=char['coords']
            coords = torch.cat([coords.long(),torch.LongTensor([idx]).expand([coords.size(0),1])],1)
            locations.append(coords)
            f=char['features']
            f = torch.cat([f.float(),torch.Tensor([idx]).expand([f.size(0),1])],1)
            features.append(f)
        return {'input': scn.InputLayerInput(torch.cat(locations,0), torch.cat(features,0)), 'target': torch.LongTensor(targets)}
    return merge

def get_iterators(*args):
    return {'train': torch.utils.data.DataLoader(Data('train.txt'), collate_fn=MergeFn(), batch_size=100,shuffle=True, ),
            'val': torch.utils.data.DataLoader(Data('test.txt'), collate_fn=MergeFn(), batch_size=100, shuffle=True,)}
