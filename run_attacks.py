"""
Script to run an attack in the experiments for “Complex network effects on the
robustness of graph convolutional networks” by Benjamin A. Miller, Kevin Chan,
and Tina Eliassi-Rad, in Applied Network Science, vol. 9, article no. 5, 2024.

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
sys.path.append('../DeepRobust/')  #path for DeepRobust code

from deeprobust.graph import defense
from deeprobust.graph.data.dataset import Dataset

from deeprobust.graph.targeted_attack.sga import SGAttack

from deeprobust.graph.targeted_attack.nettack import Nettack
from deeprobust.graph.targeted_attack.ig_attack import IGAttack

from deeprobust.graph.defense.gcn import GCN
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


dataset = sys.argv[1]
splitType = sys.argv[2]
directString = sys.argv[3]
targetString = sys.argv[4]
unlabeled_share = float(sys.argv[5])
defense = sys.argv[6]
attack = sys.argv[7]
feature_ind = int(sys.argv[8])
seeds = [104, 110, 116, 122, 128, 105, 111, 117, 123, 129, 106, 112, 118, 124, 130, 107, 113, 119, 125, 131]

real_datasets =  ['cora', 'citeseer', 'polblogs', 'pubmed']
synth_datasets = ['er', 'ba', 'ws', 'lfr', 'config', 'mag']
assert(dataset in ['cora', 'citeseer', 'polblogs', 'pubmed' 'er', 'ba', 'ws', 'lfr', 'config']), "bad dataset name"
assert(splitType in ['cover', 'degree', 'random']), "bad split string"
assert(directString in ['direct', 'indirect']), "bad direct/indirect string"
assert(targetString in ['struct', 'feat', 'both']), "bad stucture/feature string"
assert(attack in ['SGAttack', 'Nettack', 'FGA', 'IGAttack']), "bad attack string"
assert(not (defense=='SGC' and attack!='SGAttack')), "SGC only attacked by SGAttack"
assert(not (defense!='SGC' and attack=='SGAttack')), "SGAttack only attacks SGC"

direct_attack=(directString=='direct')
attack_structure = (targetString in ['struct', 'both'])
attack_features = (targetString in ['feat', 'both'])

if direct_attack:
    from deeprobust.graph.targeted_attack.fga import FGA
    from deeprobust.graph.targeted_attack.ig_attack import IGAttack
else:
    from fga_indirect import FGA_influence as FGA
    from iga_indirect import IGAttack_influence as IGAttack


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
retrain_iters = 5

for seed in seeds:
    attackedData = {}
    for _ in range(retrain_iters):
        print("... {}/{} ".format(_+1, retrain_iters))
        
        # load target data
        if dataset in real_datasets:
            target_filename='./probes/'+dataset+'_'+defense+'_'+attack+'_'+splitType+'Training_'+str(int(100*unlabeled_share))+'pctUnlabeled.pkl'
        elif feature_ind==0:
            target_filename='./probes/'+dataset+'_'+str(seed)+'_'+defense+'_'+attack+'_'+splitType+'Training_'+str(int(100*unlabeled_share))+'pctUnlabeled.pkl'
        else:
            target_filename='./probes/'+dataset+'_'+str(seed)+'_feat'+str(feature_ind)+'_'+defense+'_'+attack+'_'+splitType+'Training_'+str(int(100*unlabeled_share))+'pctUnlabeled.pkl'

        with open(target_filename, 'rb') as f:
            targetData = pkl.load(f)
        default_filename='./defaults/default_'+dataset+'_'+splitType+'_'+defense+'_'+attack+'.json'
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

        #get train/test/val split
        data = targetData[_]['dataset']
        nodeList = targetData[_]['targetList']
        #data = targetData[0]['dataset']


        val_share = 0.1
        train_share = 1 - unlabeled_share - val_share
        np.random.seed(seed)
        degrees = data.adj.sum(0).A1
        N = data.adj.shape[0]
        nClasses = data.labels.max().item() + 1
        nFeats = data.features.shape[1]
        classMat = sp.csr_matrix((np.ones((N)), (np.arange(N), data.labels)))


        # train surrogate and get weights
        attackedData[_] = {}
        if defense=='SGC':
            surrogate_model = SGC(nfeat=nFeats,
                                  nclass=nClasses,
                                  K=num_hops,
                                  cached=False,
                                  lr=lr,
                                  weight_decay=weight_decay,
                                  device=device)
            pyg_data = Dpr2Pyg(data)
            surrogate_model.train()
            surrogate_model.fit(pyg_data, patience=30)
        else:
            surrogate_model = GCN(nfeat=nFeats,
                                  nhid=nhid,
                                  nclass=nClasses,
                                  with_relu=False,
                                  dropout=dropout,
                                  lr=lr,
                                  weight_decay=weight_decay,
                                  device=device)
            surrogate_model.train()
            surrogate_model.fit(features=data.features, adj=data.adj, labels=data.labels, idx_train=data.idx_train, idx_val=data.idx_val, patience=30)


        n_influencers = 1 if direct_attack else 5 #int(min(degrees[u], 5))
        n_pert = 20 + 30*(1-direct_attack)
        for u in nodeList:
            attackedData[_][u] = {}
            if attack=='Nettack':
                attackObj = Nettack(surrogate_model, attack_structure=attack_structure, attack_features=attack_features, device=device)
                attackObj.attack(data.features, data.adj, data.labels, target_node=u, n_perturbations=n_pert, direct=direct_attack, n_influencers=n_influencers)
            elif attack=='IGAttack':
                attackObj = IGAttack(surrogate_model, nnodes=data.adj.shape[0], attack_structure=attack_structure, attack_features=attack_features, device=device)
                attackObj.attack(data.features, data.adj, data.labels, data.idx_train, target_node=u, n_perturbations=n_pert)
            elif attack=='FGA':
                attackObj = FGA(surrogate_model, nnodes=data.adj.shape[0], attack_structure=attack_structure, attack_features=attack_features, device=device)
                attackObj.attack(data.features, data.adj, data.labels, data.idx_train, target_node=u, n_perturbations=n_pert)
            else:
                attackObj = SGAttack(surrogate_model, attack_structure=attack_structure, attack_features=attack_features, device=device)
                attackObj.attack(data.features, data.adj.astype(float), data.labels, target_node=u, n_perturbations=n_pert, direct=direct_attack, n_influencers=n_influencers)


            attackedData[_][u]['feat_pert'] = copy.copy(attackObj.feature_perturbations)
            attackedData[_][u]['struct_pert'] = copy.copy(attackObj.structure_perturbations)
            del attackObj

            print(f'done with iteration {_}, {attack} against node {u}', flush=True)
        
    if dataset in real_datasets:
        filename='./attackedData/'+dataset+'_'+defense+'_'+attack+'_'+splitType+'Training_'+directString+'Attack_'+targetString+'Pert_'+str(int(100*unlabeled_share))+'pctUnlabeled.pkl'
    elif feature_ind==0:
        filename='./attackedData/'+dataset+'_'+str(seed)+'_'+defense+'_'+attack+'_'+splitType+'Training_'+directString+'Attack_'+targetString+'Pert_'+str(int(100*unlabeled_share))+'pctUnlabeled.pkl'
    else:
        filename='./attackedData/'+dataset+'_'+str(seed)+'_feat'+str(feature_ind)+'_'+defense+'_'+attack+'_'+splitType+'Training_'+directString+'Attack_'+targetString+'Pert_'+str(int(100*unlabeled_share))+'pctUnlabeled.pkl'
    with open(filename, 'wb') as f:
        pkl.dump(attackedData, f)

