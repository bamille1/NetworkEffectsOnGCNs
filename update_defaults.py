"""
Script to update hyperparameters values to their current best values for the
experiments in “Complex network effects on the robustness of graph
convolutional networks” by Benjamin A. Miller, Kevin Chan, and Tina
Eliassi-Rad, in Applied Network Science, vol. 9, article no. 5, 2024.

This material is based upon work supported by the United States Air Force under
Air  Force  Contract  No.  FA8702-15-D-0001  and  the  Combat  Capabilities
Development Command Army Research Laboratory (under Cooperative Agreement Number
W911NF-13-2-0045).  Any  opinions,  findings,  conclusions  or  recommendations
expressed in this material are those of the authors and do not necessarily
reflect theviews of the United States Air Force or Army Research Laboratory.

Copyright (C) 2023
Benjamin A. Miller and Tina Eliassi-Rad

The software is provided to you on an As-Is basis

Delivered to the U.S. Government with Unlimited Rights, as defined in DFARS Part
252.227-7013 or 7014 (Feb 2014). Notwithstanding any copyright notice, U.S.
Government rights in this work are defined by DFARS 252.227-7013 or DFARS
252.227-7014 as detailed above. Use of this work other than as specifically
authorized by the U.S. Government may violate any copyrights that exist in this
work
"""

import sys
sys.path.append('../DeepRobust/')  # path to DeepRobust code

from deeprobust.graph.data.dataset import Dataset
from deeprobust.graph.defense.chebnet import ChebNet
from deeprobust.graph.defense.gat import GAT
from deeprobust.graph.defense.gcn import GCN
from deeprobust.graph.defense.gcn_preprocess import GCNSVD
from deeprobust.graph.defense.gcn_preprocess import GCNJaccard
from deeprobust.graph.defense.median_gcn import MedianGCN
from deeprobust.graph.defense.r_gcn import RGCN
from deeprobust.graph.defense.sgc import SGC

from deeprobust.graph.data.pyg_dataset import Dpr2Pyg

import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score

import pickle as pkl
import numpy as np
import scipy.sparse as sp
import networkx as nx
import copy
import json

import os
import re



dataset = sys.argv[1]
splitType = sys.argv[2]
unlabeled_share = 0.8
defense = sys.argv[3]
attack = sys.argv[4]

seed = 100

files = os.listdir('./tempfiles/')
real_datasets =  ['cora', 'citeseer', 'polblogs', 'pubmed']

resultsDict = {}
for filename in files:
    if filename.startswith(dataset+'_'+splitType+'_'+defense+'_'+attack+'_') or filename.startswith(dataset+'_'+str(seed)+'_'+splitType+'_'+defense+'_'+attack+'_'):
        if dataset in real_datasets:
            x = re.search(r'^'+dataset+'_'+splitType+'_'+defense+'_'+attack+'_nHid(\S+)_dropout(\S+)_lr(\S+)_wd(\S+)_hops(\S+)_heads(\S+)_outheads(\S+)_gamma(\S+)_beta1(\S+)_beta2(\S+)_svd(\S+)_threshold(\S+).pkl\Z', filename)
        else:
            x = re.search(r'^'+dataset+'_'+str(seed)+'_'+splitType+'_'+defense+'_'+attack+'_nHid(\S+)_dropout(\S+)_lr(\S+)_wd(\S+)_hops(\S+)_heads(\S+)_outheads(\S+)_gamma(\S+)_beta1(\S+)_beta2(\S+)_svd(\S+)_threshold(\S+).pkl\Z', filename)
        y = tuple([x[i] for i in range(1, 13)])
        with open('./tempfiles/'+filename, 'rb') as f:
            (classResults, attackResults) = pkl.load(f)
        medianF1 = np.median([r[1] for r in classResults])
        medianMarginAfter = np.median([r[1] for r in attackResults])
        resultsDict[y] = (medianF1, medianMarginAfter)
        print(f"key: {y}--{medianF1}, {medianMarginAfter}")

maxKey = None
maxF1 = 0
maxMargin = -np.inf
for x in resultsDict:
    if (resultsDict[x][0]*100+resultsDict[x][1]*2 > maxF1*100+maxMargin*2) and (not np.isinf(resultsDict[x][0]*100+resultsDict[x][1]*2)):
        maxKey = x
        maxF1 = resultsDict[x][0]
        maxMargin = resultsDict[x][1]


default_filename = '../defaults/default_'+dataset+'_'+splitType+'_'+defense+'_'+attack+'.json'
with open(default_filename, 'r') as f:
    defaults = json.load(f)


print(f"{dataset}, {splitType}, {defense}")

print(f"current default: {defaults}")
if 'max_val' in defaults:
    old_max = defaults['max_val']

if dataset in real_datasets:
    default_pkl = dataset+'_'+splitType+'_'+defense+'_'+attack+'_nHid'+str(defaults['nhid'])+'_dropout'+str(defaults['dropout'])+'_lr'+str(defaults['lr'])+'_wd'+str(defaults['weight_decay'])+'_hops'+str(defaults['num_hops'])+'_heads'+str(defaults['heads'])+'_outheads'+str(defaults['output_heads'])+'_gamma'+str(defaults['gamma'])+'_beta1'+str(defaults['beta1'])+'_beta2'+str(defaults['beta2'])+'_svd'+str(defaults['svd_dim'])+'_threshold'+str(defaults['thres'])+'.pkl'
else:
    default_pkl = dataset+'_'+str(seed)+'_'+splitType+'_'+defense+'_'+attack+'_nHid'+str(defaults['nhid'])+'_dropout'+str(defaults['dropout'])+'_lr'+str(defaults['lr'])+'_wd'+str(defaults['weight_decay'])+'_hops'+str(defaults['num_hops'])+'_heads'+str(defaults['heads'])+'_outheads'+str(defaults['output_heads'])+'_gamma'+str(defaults['gamma'])+'_beta1'+str(defaults['beta1'])+'_beta2'+str(defaults['beta2'])+'_svd'+str(defaults['svd_dim'])+'_threshold'+str(defaults['thres'])+'.pkl'

try:
    with open('./tempfiles/'+default_pkl, 'rb') as f:
        default_class, default_attack = pkl.load(f)
        default_f1 = np.median([r[1] for r in default_class])
        default_margin = np.median([r[1] for r in default_attack])
except:
    print('no default!')
    default_f1 = 0
    default_margin = 0

if 'max_val' not in defaults:
    old_max = default_f1*100+default_margin*2
if maxKey is not None:
    allSame = (defaults['nhid'] == int(maxKey[0]) and defaults['dropout'] == float(maxKey[1]) and defaults['lr'] == float(maxKey[2]) and defaults['weight_decay'] == float(maxKey[3]) and defaults['num_hops'] == int(maxKey[4]) and defaults['heads'] == int(maxKey[5]) and defaults['output_heads'] == int(maxKey[6]) and defaults['gamma'] == float(maxKey[7]) and defaults['beta1'] == float(maxKey[8]) and defaults['beta2'] == float(maxKey[9]) and defaults['svd_dim'] == int(maxKey[10]) and defaults['thres'] == float(maxKey[11]))
    defaults['nhid'] = int(maxKey[0])
    defaults['dropout'] = float(maxKey[1])
    defaults['lr'] = float(maxKey[2])
    defaults['weight_decay'] = float(maxKey[3])
    defaults['num_hops'] = int(maxKey[4])
    defaults['heads'] = int(maxKey[5])
    defaults['output_heads'] = int(maxKey[6])
    defaults['gamma'] = float(maxKey[7])
    defaults['beta1'] = float(maxKey[8])
    defaults['beta2'] = float(maxKey[9])
    defaults['svd_dim'] = int(maxKey[10])
    defaults['thres'] = float(maxKey[11])
    defaults['max_val'] = maxF1*100+maxMargin*2
else:
    allSame = False


print(f"new default: {defaults}")
if allSame:
    print('defaults remained the same')
elif old_max >= maxF1*100+maxMargin*2:
    print(f'nothing exceeded previous max ({old_max})')
else:
    pctIncrease = (100*maxF1+2*maxMargin)/(100*default_f1+2*default_margin)
    pctIncrease = 100*pctIncrease - 100
    print(f"defaults changed: increased {pctIncrease}% from {100*default_f1+2*default_margin} to {100*maxF1+2*maxMargin}")
    with open(default_filename, 'w') as f:
        json.dump(defaults, f)
print('\n\n')
print('done')
