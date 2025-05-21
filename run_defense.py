"""
Script to run a defense method (node classifier) in the experiments for
"Complex network effects on the robustness of graph convolutional networks” by
Benjamin A. Miller, Kevin Chan, and Tina Eliassi-Rad, in Applied Network
Science, vol. 9, article no. 5, 2024.

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
sys.path.append('../DeepRobust/') #path to DeepRobust code

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


dataset = sys.argv[1]
splitType = sys.argv[2]
directString = sys.argv[3]
targetString = sys.argv[4]
unlabeled_share = float(sys.argv[5])
defense = sys.argv[6]
attack = sys.argv[7]
feature_ind = int(sys.argv[8])
seeds = [104, 110, 116, 122, 128, 105, 111, 117, 123, 129, 106, 112, 118, 124, 130, 107, 113, 119, 125, 131]

assert(dataset in ['cora', 'citeseer', 'polblogs', 'pubmed' 'er', 'ba', 'ws', 'lfr', 'config']), "bad dataset name"
assert(splitType in ['cover', 'degree', 'random']), "bad split string"
assert(directString in ['direct', 'indirect']), "bad direct/indirect string"
assert(targetString in ['struct', 'feat', 'both']), "bad stucture/feature string"
assert(defense in ['GCN', 'Jaccard', 'SVD', 'Cheb', 'median', 'GAT', 'RGCN', 'SGC']), "bad defense string"
real_datasets = ['cora', 'citeseer', 'polblogs', 'pubmed']

direct_attack=(directString=='direct')
attack_structure = (targetString in ['struct', 'both'])
attack_features = (targetString in ['feat', 'both'])
for seed in seeds:
    if dataset in real_datasets:
        infilename='./attackedData/'+dataset+'_'+defense+'_'+attack+'_'+splitType+'Training_'+directString+'Attack_'+targetString+'Pert_'+str(int(100*unlabeled_share))+'pctUnlabeled.pkl'
    elif feature_ind==0:
        infilename='./attackedData/'+dataset+'_'+str(seed)+'_'+defense+'_'+attack+'_'+splitType+'Training_'+directString+'Attack_'+targetString+'Pert_'+str(int(100*unlabeled_share))+'pctUnlabeled.pkl'
    else:
        infilename='./attackedData/'+dataset+'_'+str(seed)+'_feat'+str(feature_ind)+'_'+defense+'_'+attack+'_'+splitType+'Training_'+directString+'Attack_'+targetString+'Pert_'+str(int(100*unlabeled_share))+'pctUnlabeled.pkl'
    try:
        with open(infilename, 'rb') as f:
            attackData = pkl.load(f)
    except:
        print("Couldn't open file "+infilename)
        sys.exit()
    
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
    
    
    #data = SyntheticDataset(root='/tmp/', name=dataset, seed=seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    retrain_iters = 5
    
    if dataset in real_datasets:
        target_filename='./probes/'+dataset+'_'+defense+'_'+attack+'_'+splitType+'Training_'+str(int(100*unlabeled_share))+'pctUnlabeled.pkl'
    elif feature_ind==0:
        target_filename='./probes/'+dataset+'_'+str(seed)+'_'+defense+'_'+attack+'_'+splitType+'Training_'+str(int(100*unlabeled_share))+'pctUnlabeled.pkl'
    else:
        target_filename='./probes/'+dataset+'_'+str(seed)+'_feat'+str(feature_ind)+'_'+defense+'_'+attack+'_'+splitType+'Training_'+str(int(100*unlabeled_share))+'pctUnlabeled.pkl'
    with open(target_filename, 'rb') as f:
        targetData = pkl.load(f)
    data = targetData[0]['dataset']
    
    np.random.seed(seed)
    degrees = data.adj.sum(0).A1
    N = data.adj.shape[0]
    nClasses = data.labels.max().item() + 1
    nFeats = data.features.shape[1]
    classMat = sp.csr_matrix((np.ones((N)), (np.arange(N), data.labels)))
    
    
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
    
    
    allResults = []
    for _ in range(retrain_iters):
        print("... {}/{} ".format(_+1, retrain_iters))
        data = copy.deepcopy(targetData[_]['dataset'])
        
        
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



        #get margins for unlabeled nodes
        classification_margins_clean = np.zeros(N)
        model.eval()
        with torch.no_grad():
            probs_before_attack = np.exp(model.predict().detach().numpy())

        for n in data.idx_test:
            prob_before = probs_before_attack[n]
            
            best_second_class_before = (prob_before - 1000*classMat[n]).argmax()
            margin_before = np.log(prob_before[data.labels[n]]) - np.log(prob_before[best_second_class_before])
            classification_margins_clean[n] = margin_before
        ind = np.where(classification_margins_clean > 0)[0]


        #test
        result = {}
        result['prob_before'] = probs_before_attack
        test_true = data.labels[data.idx_test]
        test_pred = probs_before_attack[data.idx_test, :].argmax(axis=1)
        result['accuracy'] = accuracy_score(test_true, test_pred)
        result['f1_score'] = (f1_score(test_true, test_pred, average='micro'), f1_score(test_true, test_pred, average='macro'))
        result['eval_ind'] = data.idx_test

        #for attack in attacks:
        for u in targetData[_]['targetList']:
            result[u] = {}

        
            margin_after = dict()
            eval_after = dict()
            pertData = copy.deepcopy(data)
            print(f"{u}:  {attackData[_].keys()}")
            for d in range(len(attackData[_][u]['struct_pert'])):
                if attackData[_][u]['struct_pert'][d]:
                    edgeInd = attackData[_][u]['struct_pert'][d]
                    pertData.adj[edgeInd] = 1-pertData.adj[edgeInd]
                    pertData.adj[(edgeInd[1], edgeInd[0])] = 1-pertData.adj[(edgeInd[1], edgeInd[0])]
                else:
                    featInd = attackData[_][u]['feat_pert'][d]
                    pertData.features[featInd] = 1-pertData.features[featInd]

                model.train()
                if defense=='SVD':
                    model.fit(features=pertData.features, adj=pertData.adj, labels=pertData.labels, idx_train=pertData.idx_train, idx_val=pertData.idx_val, k=svd_dim, train_iters=500, patience=30)
                elif defense=='Jaccard':
                    model.fit(features=pertData.features, adj=pertData.adj, labels=pertData.labels, idx_train=pertData.idx_train, idx_val=pertData.idx_val, threshold=thres, train_iters=500, patience=30)
                elif defense in ['GCN', 'RGCN']:
                    model.fit(features=pertData.features, adj=pertData.adj, labels=pertData.labels, idx_train=pertData.idx_train, idx_val=pertData.idx_val, train_iters=500, patience=30)
                else:
                    pyg_data = Dpr2Pyg(pertData)
                    model.fit(pyg_data,  train_iters=500, patience=30)

                model.eval()
                with torch.no_grad():
                    probs_after_attack = np.exp(model.predict().detach().numpy())
                prob_after = probs_after_attack[u]
                eval_after[d+1] = prob_after.argmax()
                
                best_second_class_after = (prob_after - 1000*classMat[u]).argmax()
                margin_after[d+1] = np.log(prob_after[pertData.labels[u]]) - np.log(prob_after[best_second_class_after])
                if margin_after[d+1] < 0:
                    break
            result[u]['margin_after'] = margin_after
            result[u]['eval_after'] = eval_after

            result[u]['margin_before'] = classification_margins_clean[u]
            #result[u]['avgTrainNeighbors'] = avg_num_train_neighbors
            print(f'done with iteration {_}, {attack} against node {u}', flush=True)
            
        allResults.append(result)
    
    if dataset in real_datasets:
        filename = './defendedData/'+dataset+'_'+splitType+'Training_'+directString+'Attack_'+defense+'Defense_'+attack+'Attack_'+targetString+'Pert_'+str(int(100*unlabeled_share))+'pctUnlabeled.pkl'
    elif feature_ind==0:
        filename = './defendedData/'+dataset+'_'+str(seed)+'_'+splitType+'Training_'+directString+'Attack_'+defense+'Defense_'+attack+'Attack_'+targetString+'Pert_'+str(int(100*unlabeled_share))+'pctUnlabeled.pkl'
    else:
        filename = './defendedData/'+dataset+'_'+str(seed)+'_feat'+str(feature_ind)+'_'+splitType+'Training_'+directString+'Attack_'+defense+'Defense_'+attack+'Attack_'+targetString+'Pert_'+str(int(100*unlabeled_share))+'pctUnlabeled.pkl'
    with open(filename, 'wb') as f:
        pkl.dump(allResults, f)

