"""
Class to create synthetic datasets for experiments in “Complex network effects
on the robustness of graph convolutional networks” by Benjamin A. Miller, Kevin
Chan, and Tina Eliassi-Rad, in Applied Network Science, vol. 9, article no. 5,
2024. Derived from the "Dataset" class from DeepRobust.

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


import networkx as nx
import numpy as np
import numpy.random as rand
from scipy.sparse import linalg as spla
import scipy.sparse as sp

import sys
import os.path as osp
import re

sys.path.append('../DeepRobust/')
from deeprobust.graph.data.dataset import Dataset


class SyntheticDataset(Dataset):
    def _get_equal_split(self, v):
        N = len(v)
        mid = int(np.round(N/2))
        x = np.zeros(v.shape)
        ind = np.argsort(v)
        print(ind)
        x[ind[:mid]] = -1
        x[ind[mid:]] = 1

        return x
    
    def _swap2(self, A, x, increase=False):
        degDiff = A@x

        if increase:
            p = -degDiff*(x > 0)*(degDiff < 0)
            a = rand.choice(len(x), p=p/np.sum(p))
            p = degDiff*(x < 0)*(degDiff >= 0)
            b = rand.choice(len(x), p=p/np.sum(p))
            x[a] = -1
            x[b] = 1
        else:
            p = degDiff*(x > 0)*(degDiff >= 0)
            a = rand.choice(len(x), p=p/np.sum(p))
            p = -degDiff*(x < 0)*(degDiff < 0)
            b = rand.choice(len(x), p=p/np.sum(p))
            x[a] = -1
            x[b] = 1
    
    def __init__(self, root, name, setting='nettack', seed=None, require_mask=False,
                 feature_category=0
                ):
        self.name = name.lower()
        self.setting = setting.lower()

        assert (self.name in ['er', 'ba', 'ws', 'config']) or (self.name.startswith('lfr')), \
                'Currently only support cora, citeseer, cora_ml, ' + \
                'polblogs, pubmed, acm, blogcatalog, flickr'
        
        assert self.setting in ['gcn', 'nettack', 'prognn'], "Settings should be" + \
                        " choosen from ['gcn', 'nettack', 'prognn']"

        self.seed = seed
        # self.url =  'https://raw.githubusercontent.com/danielzuegner/nettack/master/data/%s.npz' % self.name
        #self.url =  'https://raw.githubusercontent.com/danielzuegner/gnn-meta-attack/master/data/%s.npz' % self.name
        self.url = None
        self.root = osp.expanduser(osp.normpath(root))
        self.data_folder = osp.join(root, self.name)
        self.data_filename = self.data_folder + '.npz'
        self.require_mask = require_mask
        self.feature_category = feature_category
        
        assert self.feature_category in [0, 1, 2, 3], "invalid feature category"

        self.require_lcc = False if setting == 'gcn' else True
        self.adj, self.features, self.labels = self.load_data()

        if setting == 'prognn':
            assert name in ['cora', 'citeseer', 'pubmed', 'cora_ml', 'polblogs'], "ProGNN splits only " + \
                        "cora, citeseer, pubmed, cora_ml, polblogs"
            self.idx_train, self.idx_val, self.idx_test = self.get_prognn_splits()
        else:
            self.idx_train, self.idx_val, self.idx_test = self.get_train_val_test()
        if self.require_mask:
            self.get_mask()

    def load_data(self):
        print('Loading {} dataset...'.format(self.name))
        if (self.name in ['er', 'ba', 'ws', 'kron', 'bter', 'mag', 'config']) or self.name.startswith('lfr'):
            return self.load_generated_graph()
        else:
            return None
    
    def load_generated_graph(self):
        if self.seed is  None:
            raise Exception('Must set seed to use generated graph')
        
        #set up attribute distributions
        if self.feature_category in [1, 2, 3]:
            prob=[]
            #prob.append(0.7943164854417508*.98**np.abs(np.arange(-50, 50)))
            #shift_amt = [0, 6, 14]
            prob.append(0.8080736488355945*.9**np.abs(np.arange(-10, 10)))
            shift_amt = [0, 3, 7]
            prob.append(np.roll(prob[0], shift_amt[self.feature_category-1]))
        
        # get the graph
        genSeed =83*self.seed+91
        if self.name == 'er':
            G = nx.erdos_renyi_graph(1200, 1/120, seed=genSeed)
            lcc = max(nx.connected_components(G), key=len)
            G = G.subgraph(lcc)
        elif self.name == 'ba':
            G = nx.barabasi_albert_graph(1200, 5, seed=genSeed)
        elif self.name == 'ws':
            G = nx.watts_strogatz_graph(1200, 10, .1, seed=genSeed)
            lcc = max(nx.connected_components(G), key=len)
            G = G.subgraph(lcc)
        elif self.name == 'config':
            done = False
            ctr = 0
            while not done:
                try:
                    G = nx.LFR_benchmark_graph(1200, 3, 2, .2, average_degree=10, max_degree=135, min_community=10, seed=genSeed+ctr)
                    done = True
                except:
                    ctr += 1
                    if ctr >= 83:
                        print('tried too many times')
                        raise
                    print("trying graph generation again")
            G.remove_edges_from(nx.selfloop_edges(G))
            d = dict(G.degree()).values()
            G = nx.configuration_model(d, create_using=nx.Graph, seed=genSeed)
            G.remove_edges_from(nx.selfloop_edges(G))
            lcc = max(nx.connected_components(G), key=len)
            G = G.subgraph(lcc)
        elif self.name == 'lfr':
            done = False
            ctr = 0
            while not done:
                try:
                    G = nx.LFR_benchmark_graph(1200, 3, 2, .2, average_degree=10, max_degree=135, min_community=10, seed=genSeed+ctr)
                    done = True
                except:
                    ctr += 1
                    if ctr >= 83:
                        print('tried too many times')
                        raise
                    print("trying graph generation again")
            G.remove_edges_from(nx.selfloop_edges(G))
            lcc = max(nx.connected_components(G), key=len)
            G = G.subgraph(lcc)
        else:
            raise Exception('invalid graph generator name')
        A = nx.adjacency_matrix(G).astype(np.float32)
        N = G.number_of_nodes()
        
        #get the labels
        labelType = self.seed%6
        L = nx.normalized_laplacian_matrix(G)
        if labelType in [0, 1]:
            wHi, vHi = spla.eigsh(L, k=1)
            x = self._get_equal_split(np.squeeze(vHi[:, 0]))
            minValue = x.transpose()@A@x
            if labelType == 1:
                while x.transpose()@A@x < minValue/2:
                    self._swap2(A, x, increase=True)
            labels = (x > 0).astype(int)
        else: #labelType 2, 3, 4, or 5
            wLo, vLo = spla.eigsh(L, k=2, which='SM')
            x = self._get_equal_split(vLo[:, 1])
            maxValue = x.transpose()@A@x
            while x.transpose()@A@x > (labelType-2)*maxValue/3:
                self._swap2(A, x, increase=False)
            labels = (x > 0).astype(int)
        #get the attributes
        if self.feature_category > 0:
            features = np.zeros((N, 20))
            for i in range(N):
                features[i, :] = (rand.rand(20) < prob[labels[i]])
            features = sp.csr_matrix(features, dtype=np.float32)
            features = sp.hstack((sp.eye(N, dtype=np.float32), features))
        else:
            features = sp.eye(N, dtype=np.float32)
                
        return A, features, labels
