"""
Definition of the alternative training data selection methods for the
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

import numpy as np
import networkx as nx


def degreeSplit(d, z, train_share, val_share, test_share):
    K = len(np.unique(z))
    N = len(d)
    #build the training set
    trainSet = []
    for k in range(K):
        ind = np.where(z==k)[0] #indices
        #get indices in class with higest degree and add to training set
        indInd = np.argsort(-d[ind])
        indInd = indInd[:int(np.round(len(indInd)*train_share))]
        trainSet += [ind[i] for i in indInd]
    
    testAndValSets = np.setdiff1d(np.arange(N), trainSet)
    
    return trainSet, testAndValSets

def stratifiedGreedyCover(A, z, train_share):
    G = nx.Graph(A)
    C = np.unique(z)
    K = len(C)
    N = G.number_of_nodes()
    
    
    nx.set_node_attributes(G, values=0, name='trainedNeighbors')
    trainSet = []
    candidateNodes = [[] for c in C]
    for n in range(N):
        idx = np.searchsorted(C, z[n])
        candidateNodes[idx].append(n)
    
    for idx in range(K):
        np.random.shuffle(candidateNodes[idx])
    nTrain = np.array([int(np.round(len(candidateNodes[k])*train_share)) for k in range(K)])
    remainingTrain = np.copy(nTrain)
    print(remainingTrain)
    lowestLabel = 0
    
    while np.sum(remainingTrain) > 0:
        #pick a label to use
        ratio = remainingTrain/nTrain
        classIdx = np.where(ratio==np.max(ratio))[0]
        classIdx = np.random.choice(classIdx)
        
        maxCount = -1
        maxNode = None
        for n in candidateNodes[classIdx]:
            nCount = len([u for u in G.neighbors(n) if G.nodes[u]['trainedNeighbors']==lowestLabel])
            if nCount > maxCount:
                maxNode = n
                maxCount = nCount
        if maxNode is None:
            lowestLabel += 1
        else:
            trainSet += [maxNode]
            candidateNodes[classIdx].remove(maxNode)
            remainingTrain[classIdx] -= 1
            G.nodes[maxNode]['trainedNeighbors'] = -1
            for u in G.neighbors(maxNode):
                G.nodes[u]['trainedNeighbors'] += 1

    testAndValSets = np.setdiff1d(np.arange(N), trainSet)

    return trainSet, testAndValSets


