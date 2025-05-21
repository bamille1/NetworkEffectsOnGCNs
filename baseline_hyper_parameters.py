"""
Script to tune hyperparameters for the experiments in “Complex network effects
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
Benjamin A. Miller [1], Kevin Chan [2], and Tina Eliassi-Rad [1]

[1] Northeastern Univeristy
[2] US Army Research Laboratory

The software is provided to you on an As-Is basis

Delivered to the U.S. Government with Unlimited Rights, as defined in DFARS Part
252.227-7013 or 7014 (Feb 2014). Notwithstanding any copyright notice, U.S.
Government rights in this work are defined by DFARS 252.227-7013 or DFARS
252.227-7014 as detailed above. Use of this work other than as specifically
authorized by the U.S. Government may violate any copyrights that exist in this
work
"""

import sys
sys.path.append('../DeepRobust/')  #path to DeepRobust code

from deeprobust.graph.data.dataset import Dataset
from deeprobust.graph.defense.chebnet import ChebNet
from deeprobust.graph.defense.gat import GAT
from deeprobust.graph.defense.gcn import GCN
from deeprobust.graph.defense.gcn_preprocess import GCNSVD
from deeprobust.graph.defense.gcn_preprocess import GCNJaccard
from deeprobust.graph.defense.median_gcn import MedianGCN
from deeprobust.graph.defense.r_gcn import RGCN
from deeprobust.graph.defense.sgc import SGC

from deeprobust.graph.targeted_attack.sga import SGAttack
from deeprobust.graph.targeted_attack.fga import FGA
from deeprobust.graph.targeted_attack.nettack import Nettack
from deeprobust.graph.targeted_attack.ig_attack import IGAttack

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
unlabeled_share = 0.8
defense = sys.argv[3]
attack = sys.argv[4]
nhid = int(sys.argv[5]) #default 16
dropout = float(sys.argv[6]) #default 0.5
lr = float(sys.argv[7]) #default 0.01
weight_decay = float(sys.argv[8]) #default 5e-4
num_hops = int(sys.argv[9]) #default 3, for ChebNet and SGC
heads = int(sys.argv[10]) #default 8, for GAT
output_heads = int(sys.argv[11]) #default 1, for GAT
gamma = float(sys.argv[12]) #default 1.0, for RGCN
beta1 = float(sys.argv[13]) #default 5e-4, for RGCN
beta2 = float(sys.argv[14]) #default 5e-4, for RGCN
svd_dim = int(sys.argv[15]) #default 50, for SVD
thres = float(sys.argv[16]) #defaualt 0.01, for Jaccard
seed = int(sys.argv[17])




real_datasets =  ['cora', 'citeseer', 'polblogs', 'pubmed']
synth_datasets = ['er', 'ba', 'ws', 'lfr', 'config']
assert(dataset in real_datasets+synth_datasets), "bad dataset name"

print(f"dataset: {dataset}")
print(f"split type: {splitType}")
print(f"defense: {defense}")
print(f"# hidden units: {nhid}")
print(f"dropout rate: {dropout}")
print(f"learning rate: {lr}")
print(f"weight decay: {weight_decay}")
print(f"number of hops: {num_hops}")
print(f"heads: {heads}")
print(f"output_heads: {output_heads}")
print(f"gamma: {gamma}")
print(f"beta1: {beta1}")
print(f"beta2: {beta2}")
print(f"svd dimension: {svd_dim}|")
print(f"Jaccard threshold: {thres}")


assert(svd_dim <= 500), "svd dimension too large"

assert(splitType in ['cover', 'degree', 'random']), "bad split string"
assert((unlabeled_share < 0.9) and (unlabeled_share > 0)), "bad unlabeled proportion"
assert(defense in ['GCN', 'Jaccard', 'SVD', 'Cheb', 'median', 'GAT', 'RGCN', 'SGC']), "bad defense string"

if (defense=='SGC' and attack!='SGAttack') or (defense!='SGC' and attack=='SGAttack'):
    print('SGC <==> SGAttack')
    sys.exit(0)


#create a temporary data directory
tempDir = './temp/'
#seed = 17
if dataset in ['cora', 'citeseer', 'polblogs', 'pubmed', 'acm', 'uai', 'blogcatalog']:
    data = Dataset(root=tempDir, name=dataset, seed=seed)
else:
    data = SyntheticDataset(root=tempDir, name=dataset, seed=seed)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
retrain_iters = 5
#data = Dataset(root=tempDir, name=dataset, seed=seed)

#splitType = 'cover'
#direct_attack = False
val_share = 0.1
train_share = 1 - unlabeled_share - val_share
np.random.seed(seed)





#train with no perturbations
classResults = []
attackResults = []
for _ in range(retrain_iters):
    print("... {}/{} ".format(_+1, retrain_iters))
    if dataset in synth_datasets:
        data = SyntheticDataset(root=tempDir, name=dataset, seed=(seed+_*6))
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
    
    if defense=='SGC':
        surrogate_model = SGC(nfeat=nFeats,
                              nclass=nClasses,
                              K=num_hops,
                              cached=False,
                              lr=lr,
                              weight_decay=weight_decay,
                              device=device)
    else:
        surrogate_model = GCN(nfeat=nFeats,
                              nhid=nhid,
                              nclass=nClasses,
                              with_relu=False,
                              dropout=dropout,
                              lr=lr,
                              weight_decay=weight_decay,
                              device=device)
    
    
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
    
    print(f"adj shape: {data.adj.shape}, feat shape: {data.features.shape}, label shape: {data.labels.shape}")
    #train with no perturbations
    model.train()
    if defense=='SVD':
        model.fit(features=data.features, adj=data.adj, labels=data.labels, idx_train=data.idx_train, idx_val=data.idx_val, k=svd_dim, train_iters=500, patience=30)
    elif defense=='Jaccard':
        model.fit(features=data.features, adj=data.adj, labels=data.labels, idx_train=data.idx_train, idx_val=data.idx_val, threshold=thres, train_iters=500, patience=30)
    elif defense in ['GCN', 'RGCN']:
        model.fit(features=data.features, adj=data.adj, labels=data.labels, idx_train=data.idx_train, idx_val=data.idx_val, train_iters=500, patience=30)
    else:
        pyg_data = Dpr2Pyg(data)
        model.fit(pyg_data,  train_iters=500, patience=30)
    
    surrogate_model.train()
    if defense=='SGC':
        surrogate_model.fit(pyg_data, patience=30)
    else:
        surrogate_model.fit(features=data.features, adj=data.adj, labels=data.labels, idx_train=data.idx_train, idx_val=data.idx_val, patience=30)

    model.eval()
    with torch.no_grad():
        probs_before_attack = np.exp(model.predict().detach().numpy())
    val_true = data.labels[data.idx_val]
    val_pred = probs_before_attack[data.idx_val, :].argmax(axis=1)
    classResults.append((f1_score(val_true, val_pred, average='micro'), f1_score(val_true, val_pred, average='macro')))

    #what happens if we attack?
    classification_margins_clean = np.zeros(N)

    for n in split_unlabeled:
        prob_before = probs_before_attack[n]
        best_second_class_before = (prob_before - 1000*classMat[n]).argmax()
        margin_before = np.log(prob_before[data.labels[n]]) - np.log(prob_before[best_second_class_before])
        classification_margins_clean[n] = margin_before
    ind = np.where(classification_margins_clean > 0)[0]

    nodeList = np.random.choice(ind, 10, replace=False)
    n_influencers = 1
    n_pert = 5
    for u in nodeList:
        if attack=='Nettack':
            attackObj = Nettack(surrogate_model, attack_structure=True, attack_features=False, device=device)
            attackObj.attack(data.features, data.adj, data.labels, target_node=u, n_perturbations=n_pert, direct=True, n_influencers=n_influencers)
        elif attack=='IGAttack':
            attackObj = IGAttack(surrogate_model, attack_structure=True, attack_features=False, device=device)
            attackObj.attack(data.features, data.adj, data.labels, data.idx_train, target_node=u, n_perturbations=n_pert)
        elif attack=='FGA':
            attackObj = FGA(surrogate_model, nnodes=data.adj.shape[0], attack_structure=True, attack_features=False, device=device)
            attackObj.attack(data.features, data.adj, data.labels, data.idx_train, target_node=u, n_perturbations=n_pert)
        else:
            attackObj = SGAttack(surrogate_model, attack_structure=True, attack_features=False, device=device)
            attackObj.attack(data.features, data.adj.astype(float), data.labels, target_node=u, n_perturbations=n_pert, direct=True, n_influencers=n_influencers)

        pertData = copy.deepcopy(data)
        print(f"{u}:  {attackObj.structure_perturbations}")
        for d in range(5):
            edgeInd = attackObj.structure_perturbations[d]
            pertData.adj[edgeInd] = 1-pertData.adj[edgeInd]
            pertData.adj[(edgeInd[1], edgeInd[0])] = 1-pertData.adj[(edgeInd[1], edgeInd[0])]

        model.train()
        if defense=='SVD':
            model.fit(features=pertData.features, adj=pertData.adj, labels=pertData.labels, idx_train=pertData.idx_train, idx_val=pertData.idx_val, k=svd_dim, train_iters=500, patience=30)
        elif defense=='Jaccard':
            model.fit(features=pertData.features, adj=pertData.adj, labels=pertData.labels, idx_train=pertData.idx_train, idx_val=pertData.idx_val, threshold=thres, train_iters=500, patience=30)
        elif defense in ['GCN', 'RGCN']:
            model.fit(features=pertData.features, adj=pertData.adj, labels=pertData.labels, idx_train=pertData.idx_train, idx_val=pertData.idx_val, train_iters=500, patience=30)
            #targetData[_]['dataset'] = copy.deepcopy(data)
            #targetData[_]['classifier'] = copy.deepcopy(model)
        else:
            pyg_data = Dpr2Pyg(pertData)
            model.fit(pyg_data,  train_iters=500, patience=30)
            #targetData[_]['dataset'] = copy.deepcopy(pyg_data)
            #targetData[_]['classifier'] = copy.deepcopy(model)

        model.eval()
        with torch.no_grad():
            probs_after_attack = np.exp(model.predict().detach().numpy())
        prob_after = probs_after_attack[u]
        eval_after = prob_after.argmax()

        best_second_class_after = (prob_after - 1000*classMat[u]).argmax()
        margin_after = np.log(prob_after[pertData.labels[u]]) - np.log(prob_after[best_second_class_after])
        attackResults.append((classification_margins_clean[u], margin_after))

        del attackObj

        print(f'done with iteration {_}, {attack} against node {u}', flush=True)

mean1 = np.mean([x[0] for x in classResults])
mean2 = np.mean([x[1] for x in classResults])
if dataset in real_datasets:
    filename = '../tempfiles/'+dataset+'_'+splitType+'_'+defense+'_'+attack+'_nHid'+str(nhid)+'_dropout'+str(dropout)+'_lr'+str(lr)+'_wd'+str(weight_decay)+'_hops'+str(num_hops)+'_heads'+str(heads)+'_outheads'+str(output_heads)+'_gamma'+str(gamma)+'_beta1'+str(beta1)+'_beta2'+str(beta2)+'_svd'+str(svd_dim)+'_threshold'+str(thres)+'.pkl'
else:
    filename = '../tempfiles/'+dataset+'_'+str(seed)+'_'+splitType+'_'+defense+'_'+attack+'_nHid'+str(nhid)+'_dropout'+str(dropout)+'_lr'+str(lr)+'_wd'+str(weight_decay)+'_hops'+str(num_hops)+'_heads'+str(heads)+'_outheads'+str(output_heads)+'_gamma'+str(gamma)+'_beta1'+str(beta1)+'_beta2'+str(beta2)+'_svd'+str(svd_dim)+'_threshold'+str(thres)+'.pkl'

with open(filename, "wb") as f:
    pkl.dump((classResults, attackResults), f)
