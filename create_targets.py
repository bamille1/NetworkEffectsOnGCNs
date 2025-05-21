"""
Script to identify target nodes for the experiments in “Complex network effects
on the robustness of graph convolutional networks” by Benjamin A. Miller, Kevin
Chan, and Tina Eliassi-Rad, in Applied Network Science, vol. 9, article no. 5,
2024.

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
sys.path.append('../DeepRobust/')  #path for DeepRobust code

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

from synthetic_dataset import SyntheticDataset

import pickle as pkl
import numpy as np
import scipy.sparse as sp
import networkx as nx
import copy
import json

from training_selection import degreeSplit, stratifiedGreedyCover


dataset = sys.argv[1]
splitType = sys.argv[2]
unlabeled_share = float(sys.argv[3])
defense = sys.argv[4]
attack = sys.argv[5]
if len(sys.argv) == 7:
    seeds =  [104, 110, 116, 122, 128, 105, 111, 117, 123, 129, 106, 112, 118, 124, 130, 107, 113, 119, 125, 131]
    feature_ind = int(sys.argv[6])
elif len(sys.argv) == 8:
    seeds = [int(sys.argv[7])]
    feature_ind = int(sys.argv[6])

real_datasets =  ['cora', 'citeseer', 'polblogs', 'pubmed', 'acm', 'uai', 'blogcatalog']
synth_datasets = ['er', 'ba', 'ws', 'lfr', 'config', 'mag']

#previously created temporary data directory
tempDir='./temp/'


assert(splitType in ['cover', 'degree', 'random']), "bad split string"
assert((unlabeled_share < 0.9) and (unlabeled_share > 0)), "bad unlabeled proportion"
assert(defense in ['GCN', 'Jaccard', 'SVD', 'Cheb', 'median', 'GAT', 'RGCN', 'SGC']), "bad defense string"

assert(dataset in real_datasets+synth_datasets), "bad dataset name"
default_filename = '../defaults/default_'+dataset+'_'+splitType+'_'+defense+'_'+attack+'.json'

with open(default_filename, 'r') as f:
    defaults = json.load(f)

nhid = defaults['nhid']
dropout = defaults['dropout']
lr = defaults['lr']
weight_decay = defaults['weight_decay']
num_hops = defaults['num_hops']
heads = defaults['heads']
output_heads = defaults['output_heads']
gamma = defaults['gamma']
beta1 = defaults['beta1']
beta2 = defaults['beta2']
svd_dim = defaults['svd_dim']
thres = defaults['thres']

for seed in seeds:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    retrain_iters = 5
    if dataset in real_datasets:
        data = Dataset(root=tempDir, name=dataset, seed=seed)
    else:
        data = SyntheticDataset(root=tempDir, name=dataset, seed=seed, feature_category=feature_ind)
    
    val_share = 0.1
    train_share = 1 - unlabeled_share - val_share
    np.random.seed(seed)
    degrees = data.adj.sum(0).A1
    N = data.adj.shape[0]
    nClasses = data.labels.max().item() + 1
    nFeats = data.features.shape[1]
    classMat = sp.csr_matrix((np.ones((N)), (np.arange(N), data.labels)))
    
    #remove self loops, if they exist
    data.adj = data.adj - sp.diags(data.adj.diagonal())
    
    if splitType == 'degree':
        split_train, split_val_and_test = degreeSplit(d=degrees, z=data.labels, train_share=train_share, val_share=val_share, test_share=unlabeled_share)
    elif splitType == 'cover':
        split_train, split_val_and_test = stratifiedGreedyCover(data.adj, data.labels, train_share)
    
    
    if defense=='GCN':
        model = GCN(nfeat=nFeats,
                    nhid=nhid,
                    nclass=nClasses,
                    dropout=dropout,
                    lr=lr,
                    weight_decay=weight_decay,
                    device=device)
    elif defense=='Jaccard':
        model = GCNJaccard(nfeat=nFeats,
                           nhid=nhid,
                           nclass=nClasses,
                           device=device)
    elif defense=='SVD':
        model = GCNSVD(nfeat=nFeats,
                       nhid=nhid,
                       nclass=nClasses,
                       dropout=dropout,
                       lr=lr,
                       weight_decay=weight_decay,
                       device=device)
    elif defense=='Cheb':
        model = ChebNet(nfeat=nFeats,
                        nhid=nhid,
                        nclass=nClasses,
                        num_hops=num_hops,
                        dropout=dropout,
                        lr=lr,
                        weight_decay=weight_decay,
                        device=device)
    elif defense=='median':
        model = MedianGCN(nfeat=nFeats,
                          nhid=nhid,
                          nclass=nClasses,
                          dropout=dropout,
                          lr=lr,
                          weight_decay=weight_decay,
                          device=device)
    elif defense=='GAT':
        model = GAT(nfeat=nFeats,
                    nhid=nhid,
                    nclass=nClasses,
                    heads=heads,
                    output_heads=output_heads,
                    dropout=dropout,
                    lr=lr,
                    weight_decay=weight_decay,
                    device=device)
    elif defense=='RGCN':
        model = RGCN(nnodes=data.adj.shape[0],
                     nfeat=nFeats,
                     nhid=nhid,
                     nclass=nClasses,
                     gamma=gamma,
                     beta1=beta1,
                     beta2=beta2,
                     lr=lr,
                     dropout=dropout,
                     device=device)
    elif defense=='SGC':
        model = SGC(nfeat=nFeats,
                    nclass=nClasses,
                    K=num_hops,
                    cached=False,
                    lr=lr,
                    weight_decay=weight_decay,
                    device=device)
    
    
    
    #train with no perturbations
    targetData = []
    if dataset in real_datasets:
        filename='./probes/'+dataset+'_'+defense+'_'+attack+'_'+splitType+'Training_'+str(int(100*unlabeled_share))+'pctUnlabeled.pkl'
    elif feature_ind==0:
        filename='./probes/'+dataset+'_'+str(seed)+'_'+defense+'_'+attack+'_'+splitType+'Training_'+str(int(100*unlabeled_share))+'pctUnlabeled.pkl'
    else:
        filename='./probes/'+dataset+'_'+str(seed)+'_feat'+str(feature_ind)+'_'+defense+'_'+attack+'_'+splitType+'Training_'+str(int(100*unlabeled_share))+'pctUnlabeled.pkl'
    
    for _ in range(retrain_iters):
        print("... {}/{} ".format(_+1, retrain_iters))
        targetData.append({})
        #get train/test/val split
        if splitType == 'random':
            split_train, split_val_and_test = train_test_split(np.arange(N),\
                                                               random_state=None,\
                                                               train_size=train_share,\
                                                               test_size=(val_share+unlabeled_share),\
                                                               stratify=data.labels)
        split_val, split_unlabeled = train_test_split(split_val_and_test,\
                                                      random_state=None,\
                                                      train_size=(val_share / (unlabeled_share + val_share)),\
                                                      test_size=None,\
                                                      stratify=data.labels[split_val_and_test])
        data.idx_train = np.array(split_train)
        data.idx_val = np.array(split_val)
        data.idx_test = np.array(split_unlabeled)
        
        
        #train with no perturbations
        model.train()
        if defense in ['GCN', 'SVD', 'Jaccard', 'RGCN']:
            model.fit(features=data.features, adj=data.adj, labels=data.labels, idx_train=data.idx_train, idx_val=data.idx_val, patience=30)
        else:
            pyg_data = Dpr2Pyg(data)
            model.fit(pyg_data, patience=30)

        targetData[_]['dataset'] = copy.deepcopy(data)
        classification_margins_clean = np.zeros(N)
        model.eval()
        with torch.no_grad():
            probs_before_attack = np.exp(model.predict().detach().numpy())
        
        for n in split_unlabeled:
            prob_before = probs_before_attack[n]           
            best_second_class_before = (prob_before - 1000*classMat[n]).argmax()
            margin_before = np.log(prob_before[data.labels[n]]) - np.log(prob_before[best_second_class_before])
            classification_margins_clean[n] = margin_before
        ind = np.where(classification_margins_clean > 0)[0]
        
        nodeList = np.random.choice(ind, 25, replace=False)
        targetData[_]['targetList']  = np.copy(nodeList)
    
    
    with open(filename, 'wb') as f:
        pkl.dump(targetData, f)

