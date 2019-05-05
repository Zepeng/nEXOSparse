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

def interp(sample,x,y):
    return torch.from_numpy(np.hstack([np.interp(sample.numpy(),x.numpy(),y[:,i].numpy())[:,None] for i in range(y.shape[1])])).float()
class Data(torch.utils.data.Dataset):
    def __init__(self,file,scale=63,repeats=1):
        torch.utils.data.Dataset.__init__(self)
        flist = open(file, 'r')
        for k, pfile in enumerate(flist):
            self.data = pickle.load(open(pfile.strip(), 'rb'))
            for j in range(len(self.data)):
                self.data[j]['coords'] = torch.from_numpy(self.data[j]['coords'])
                self.data[j]['features'] = torch.from_numpy(self.data[j]['features']).unsqueeze(1)
                self.data[j]['target'] = self.data[j]['target']
                self.data[j]['features'] = torch.cat([self.data[j]['features'], self.data[j]['features']], 1)
    def __getitem__(self,n):
        return self.data[n]
    def __len__(self):
        return len(self.data)

def TrainMergeFn(spatial_size=95, jitter=8):
    def merge(tbl):
        v=torch.Tensor([[1,0,0]])
        targets=[x['target'] for x in tbl]
        locations=[]
        features=[]
        for idx,char in enumerate(tbl):
            m = torch.eye(2)
            r = torch.randint(0,3,[1]).int().item()
            alpha = torch.rand(1).item()*0.4-0.2
            if r == 1:
                m[0][1] = alpha
            elif r == 2:
                m[1][0] = alpha
            else:
                m = torch.mm(m, torch.FloatTensor(
                    [[math.cos(alpha), math.sin(alpha)],
                     [-math.sin(alpha), math.cos(alpha)]]))
            coords=char['coords']
            coords = torch.cat([coords.long(),torch.LongTensor([idx]).expand([coords.size(0),1])],1)
            locations.append(coords)
            f=char['features']
            f=torch.mm(f.short(), m.short())
            #f /= (f**2).sum(1,keepdim=True)**0.5
            f = torch.cat([f.float(),torch.ones([f.size(0),1])],1)
            features.append(f)
        return {'input': scn.InputLayerInput(torch.cat(locations,0), torch.cat(features,0)), 'target': torch.LongTensor(targets)}
    return merge
def TestMergeFn(spatial_size=95):
    def merge(tbl):
        v=torch.Tensor([[1,0,0]])
        targets=[x['target'] for x in tbl]
        locations=[]
        features=[]
        for idx,char in enumerate(tbl):
            coords=char['coords']
            coords = torch.cat([coords.long(),torch.LongTensor([idx]).expand([coords.size(0),1])],1)
            locations.append(coords)
            f=char['features']
            f = torch.cat([f.float(),torch.ones([f.size(0),1])],1)
            features.append(f)
        return {'input': scn.InputLayerInput(torch.cat(locations,0), torch.cat(features,0)), 'target': torch.LongTensor(targets)}
    return merge


def get_iterators(*args):
    return {'train': torch.utils.data.DataLoader(Data('data/train.txt',repeats=1), collate_fn=TrainMergeFn(), batch_size=50, shuffle=True, num_workers=10),
            'val': torch.utils.data.DataLoader(Data('data/test.txt',repeats=1), collate_fn=TestMergeFn(), batch_size=100, shuffle=True, num_workers=10)}
