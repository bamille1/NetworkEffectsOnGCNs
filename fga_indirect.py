"""
This is a modification of the Fast Gradient Attack (FGA) (as implemented in
DeepRobust) to enable influence attacks (i.e., those that do not directly
modify the target's edges or attributes).This was used for the experiments in
“Complex network effects on the robustness of graph convolutional networks” by
Benjamin A. Miller, Kevin Chan, and Tina Eliassi-Rad, in Applied Network
Science, vol. 9, article no. 5, 2024.

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

"""
    FGA: Fast Gradient Attack on Network Embedding (https://arxiv.org/pdf/1809.02797.pdf)
    Another very similar algorithm to mention here is FGSM (for graph data).
    It is mentioned in Zugner's paper,
    Adversarial Attacks on Neural Networks for Graph Data, KDD'19
"""

import torch
import sys
sys.path.append('../DeepRobust/')
from deeprobust.graph.targeted_attack import BaseAttack
from torch.nn.parameter import Parameter
from copy import deepcopy
from deeprobust.graph import utils
import torch.nn.functional as F
import scipy.sparse as sp
from numba import jit

import numpy as np

class FGA_influence(BaseAttack):
    """FGA/FGSM.

    Parameters
    ----------
    model :
        model to attack
    nnodes : int
        number of nodes in the input graph
    feature_shape : tuple
        shape of the input node features
    attack_structure : bool
        whether to attack graph structure
    attack_features : bool
        whether to attack node features
    device: str
        'cpu' or 'cuda'

    Examples
    --------

    >>> from deeprobust.graph.data import Dataset
    >>> from deeprobust.graph.defense import GCN
    >>> from deeprobust.graph.targeted_attack import FGA
    >>> data = Dataset(root='/tmp/', name='cora')
    >>> adj, features, labels = data.adj, data.features, data.labels
    >>> idx_train, idx_val, idx_test = data.idx_train, data.idx_val, data.idx_test
    >>> # Setup Surrogate model
    >>> surrogate = GCN(nfeat=features.shape[1], nclass=labels.max().item()+1,
                    nhid=16, dropout=0, with_relu=False, with_bias=False, device='cpu').to('cpu')
    >>> surrogate.fit(features, adj, labels, idx_train, idx_val, patience=30)
    >>> # Setup Attack Model
    >>> target_node = 0
    >>> model = FGA(surrogate, nnodes=adj.shape[0], attack_structure=True, attack_features=False, device='cpu').to('cpu')
    >>> # Attack
    >>> model.attack(features, adj, labels, idx_train, target_node, n_perturbations=5)
    >>> modified_adj = model.modified_adj

    """

    def __init__(self, model, nnodes, feature_shape=None, attack_structure=True, attack_features=False, device='cpu'):

        super(FGA_influence, self).__init__(model, nnodes, attack_structure=attack_structure, attack_features=attack_features, device=device)


        assert not self.attack_features, "not support attacking features"

        if self.attack_features:
            self.feature_changes = Parameter(torch.FloatTensor(feature_shape))
            self.feature_changes.data.fill_(0)


    # pick targets with the same code used for indirect Nettack    
    def get_linearized_weight(self):
        surrogate = self.surrogate
        W = surrogate.gc1.weight @ surrogate.gc2.weight
        return W.detach().cpu().numpy()

    # from Nettack code    
    def compute_new_a_hat_uv(self, potential_edges, target_node, modified_adj, adj_norm):
        """
        Compute the updated A_hat_square_uv entries that would result from inserting/deleting the input edges,
        for every edge.
        Parameters
        ----------
        potential_edges: np.array, shape [P,2], dtype int
            The edges to check.
        Returns
        -------
        sp.sparse_matrix: updated A_hat_square_u entries, a sparse PxN matrix, where P is len(possible_edges).
        """

        edges = np.array(modified_adj.nonzero()).T
        edges_set = {tuple(x) for x in edges}
        A_hat_sq = adj_norm @ adj_norm
        values_before = A_hat_sq[target_node].toarray()[0]
        node_ixs = np.unique(edges[:, 0], return_index=True)[1]
        twohop_ixs = np.array(A_hat_sq.nonzero()).T
        degrees = modified_adj.sum(0).A1 + 1

        ixs, vals = compute_new_a_hat_uv(edges, node_ixs, edges_set, twohop_ixs, values_before, degrees,
                                         potential_edges.astype(np.int32), target_node)
        ixs_arr = np.array(ixs)
        a_hat_uv = sp.coo_matrix((vals, (ixs_arr[:, 0], ixs_arr[:, 1])), shape=[len(potential_edges), self.nnodes])

        return a_hat_uv

    def struct_score(self, a_hat_uv, XW):
        """
        Compute structure scores, cf. Eq. 15 in the paper
        Parameters
        ----------
        a_hat_uv: sp.sparse_matrix, shape [P,2]
            Entries of matrix A_hat^2_u for each potential edge (see paper for explanation)
        XW: sp.sparse_matrix, shape [N, K], dtype float
            The class logits for each node.
        Returns
        -------
        np.array [P,]
            The struct score for every row in a_hat_uv
        """

        logits = a_hat_uv.dot(XW)
        label_onehot = np.eye(XW.shape[1])[self.label_u]
        best_wrong_class_logits = (logits - 1000 * label_onehot).max(1)
        logits_for_correct_class = logits[:,self.label_u]
        struct_scores = logits_for_correct_class - best_wrong_class_logits

        return struct_scores

    def get_attacker_nodes(self, target_node, ori_adj, modified_adj, modified_features, adj_norm, n_influencers, add_additional_nodes=True):
        """Determine the influencer nodes to attack node i based on
        the weights W and the attributes X.
        """
        assert n_influencers < self.nnodes-1, "number of influencers cannot be >= number of nodes in the graph!"
        neighbors = ori_adj[target_node].nonzero()[1]
        assert target_node not in neighbors

        potential_edges = np.column_stack((np.tile(target_node, len(neighbors)),neighbors)).astype("int32")

        # The new A_hat_square_uv values that we would get if we removed the edge from u to each of the neighbors, respectively
        a_hat_uv = self.compute_new_a_hat_uv(potential_edges, target_node, modified_adj, adj_norm)

        # XW = self.compute_XW()
        W = self.get_linearized_weight()
        XW = modified_features @ W

        # compute the struct scores for all neighbors
        struct_scores = self.struct_score(a_hat_uv, XW)
        if len(neighbors) >= n_influencers:  # do we have enough neighbors for the number of desired influencers?
            influencer_nodes = neighbors[np.argsort(struct_scores)[:n_influencers]]
            if add_additional_nodes:
                return influencer_nodes, np.array([])
            return influencer_nodes
        else:

            influencer_nodes = neighbors
            if add_additional_nodes:  # Add additional influencers by connecting them to u first.
                # Compute the set of possible additional influencers, i.e. all nodes except the ones
                # that are already connected to u.
                poss_add_infl = np.setdiff1d(np.setdiff1d(np.arange(self.nnodes),neighbors), target_node)
                n_possible_additional = len(poss_add_infl)
                n_additional_attackers = n_influencers-len(neighbors)
                possible_edges = np.column_stack((np.tile(target_node, n_possible_additional), poss_add_infl))

                # Compute the struct_scores for all possible additional influencers, and choose the one
                # with the best struct score.
                a_hat_uv_additional = self.compute_new_a_hat_uv(possible_edges, target_node, modified_adj, adj_norm)
                additional_struct_scores = self.struct_score(a_hat_uv_additional, XW)
                additional_influencers = poss_add_infl[np.argsort(additional_struct_scores)[-n_additional_attackers::]]

                return influencer_nodes, additional_influencers
            else:
                return influencer_nodes


    def attack(self, ori_features, ori_adj, labels, idx_train, target_node, n_perturbations, n_influencers=5, verbose=False, **kwargs):
        """Generate perturbations on the input graph.

        Parameters
        ----------
        ori_features : scipy.sparse.csr_matrix
            Original (unperturbed) adjacency matrix
        ori_adj : scipy.sparse.csr_matrix
            Original (unperturbed) node feature matrix
        labels :
            node labels
        idx_train:
            training node indices
        target_node : int
            target node index to be attacked
        n_perturbations : int
            Number of perturbations on the input graph. Perturbations could
            be edge removals/additions or feature removals/additions.
        """
        # hold onto sparse version of matrices
        lil_adj = ori_adj.tolil()
        lil_features = ori_features.tolil()
        adj_norm = utils.normalize_adj(lil_adj)
        modified_adj = ori_adj.todense()
        modified_features = ori_features.todense()
        modified_adj, modified_features, labels = utils.to_tensor(modified_adj, modified_features, labels, device=self.device)

        self.surrogate.eval()
        if verbose == True:
            print('number of pertubations: %s' % n_perturbations)

        pseudo_labels = self.surrogate.predict().detach().argmax(1)
        pseudo_labels[idx_train] = labels[idx_train]

        self.structure_perturbations = []
        self.feature_perturbations = []
        modified_adj.requires_grad = True
        
        self.label_u = labels[target_node]  #added by b.a. miller
        infls, add_infls = self.get_attacker_nodes(target_node, ori_adj, lil_adj, lil_features, adj_norm, n_influencers, add_additional_nodes=True)
        influencers = np.concatenate((infls, add_infls)).astype("int")
        
        for i in range(n_perturbations):
            adj_norm = utils.normalize_adj_tensor(modified_adj)

            if self.attack_structure:
                output = self.surrogate(modified_features, adj_norm)
                loss = F.nll_loss(output[[target_node]], pseudo_labels[[target_node]])
                grad = torch.autograd.grad(loss, modified_adj)[0]
                # bidirection
                #grad = (grad[influencers] + grad[:, influencers]) * (-2*modified_adj[influencers] + 1)
                #grad[influencers] = -10
                #grad_argmax = torch.argmax(grad)
                grad = (grad[influencers] + grad[:, influencers].transpose(0, 1)) * (-2*modified_adj[influencers]+1)
                grad[:, target_node] = -10
                for j in range(n_influencers):
                    grad[j, influencers[j]] = -10
                grad_argmax = torch.argmax(grad)
                row_ind = (grad_argmax/grad.shape[1]).floor().type(torch.LongTensor)
                row_ind = influencers[row_ind]
                col_ind = grad_argmax%grad.shape[1]

            #print(f"    inner max before: {modified_adj.detach().cpu().numpy().max()}")
            #print(f"    entry1: {modified_adj[row_ind][col_ind]}")
            #print(f"    entry2: {modified_adj[col_ind][row_ind]}")
            #print(f"     data1: {modified_adj.data[row_ind][col_ind]}")
            #print(f"     data2: {modified_adj.data[col_ind][row_ind]}")
            value = -2*modified_adj[row_ind][col_ind] + 1
            modified_adj.data[row_ind][col_ind] += value
            modified_adj.data[col_ind][row_ind] += value
            #print(f"    inner max after: {modified_adj.detach().cpu().numpy().max()}")

            #self.structure_perturbations.append((target_node, grad_argmax))
            self.structure_perturbations.append((row_ind, col_ind))
            self.feature_perturbations.append(())

            if self.attack_features:
                pass

        modified_adj = modified_adj.detach().cpu().numpy()
        modified_adj = sp.csr_matrix(modified_adj)
        self.check_adj(modified_adj)
        self.modified_adj = modified_adj
        # self.modified_features = modified_features

@jit(nopython=True)
def connected_after(u, v, connected_before, delta):
    if u == v:
        if delta == -1:
            return False
        else:
            return True
    else:
        return connected_before


@jit(nopython=True)
def compute_new_a_hat_uv(edge_ixs, node_nb_ixs, edges_set, twohop_ixs, values_before, degs, potential_edges, u):
    """
    Compute the new values [A_hat_square]_u for every potential edge, where u is the target node. C.f. Theorem 5.1
    equation 17.
    """
    N = degs.shape[0]

    twohop_u = twohop_ixs[twohop_ixs[:, 0] == u, 1]
    nbs_u = edge_ixs[edge_ixs[:, 0] == u, 1]
    nbs_u_set = set(nbs_u)

    return_ixs = []
    return_values = []

    for ix in range(len(potential_edges)):
        edge = potential_edges[ix]
        edge_set = set(edge)
        degs_new = degs.copy()
        delta = -2 * ((edge[0], edge[1]) in edges_set) + 1
        degs_new[edge] += delta

        nbs_edge0 = edge_ixs[edge_ixs[:, 0] == edge[0], 1]
        nbs_edge1 = edge_ixs[edge_ixs[:, 0] == edge[1], 1]

        affected_nodes = set(np.concatenate((twohop_u, nbs_edge0, nbs_edge1)))
        affected_nodes = affected_nodes.union(edge_set)
        a_um = edge[0] in nbs_u_set
        a_un = edge[1] in nbs_u_set

        a_un_after = connected_after(u, edge[0], a_un, delta)
        a_um_after = connected_after(u, edge[1], a_um, delta)

        for v in affected_nodes:
            a_uv_before = v in nbs_u_set
            a_uv_before_sl = a_uv_before or v == u

            if v in edge_set and u in edge_set and u != v:
                if delta == -1:
                    a_uv_after = False
                else:
                    a_uv_after = True
            else:
                a_uv_after = a_uv_before
            a_uv_after_sl = a_uv_after or v == u

            from_ix = node_nb_ixs[v]
            to_ix = node_nb_ixs[v + 1] if v < N - 1 else len(edge_ixs)
            node_nbs = edge_ixs[from_ix:to_ix, 1]
            node_nbs_set = set(node_nbs)
            a_vm_before = edge[0] in node_nbs_set

            a_vn_before = edge[1] in node_nbs_set
            a_vn_after = connected_after(v, edge[0], a_vn_before, delta)
            a_vm_after = connected_after(v, edge[1], a_vm_before, delta)

            mult_term = 1 / np.sqrt(degs_new[u] * degs_new[v])

            sum_term1 = np.sqrt(degs[u] * degs[v]) * values_before[v] - a_uv_before_sl / degs[u] - a_uv_before / \
                        degs[v]
            sum_term2 = a_uv_after / degs_new[v] + a_uv_after_sl / degs_new[u]
            sum_term3 = -((a_um and a_vm_before) / degs[edge[0]]) + (a_um_after and a_vm_after) / degs_new[edge[0]]
            sum_term4 = -((a_un and a_vn_before) / degs[edge[1]]) + (a_un_after and a_vn_after) / degs_new[edge[1]]
            new_val = mult_term * (sum_term1 + sum_term2 + sum_term3 + sum_term4)

            return_ixs.append((ix, v))
            return_values.append(new_val)

    return return_ixs, return_values
