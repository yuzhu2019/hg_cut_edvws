import numpy as np
import networkx as nx
import random
inf = float('inf')

def clique_expansion(incidence_list, parameter_list, edge_weight, n_v, n_e, c0_labeled, c1_labeled):
    
    A = np.zeros((n_v+2, n_v+2))     # the adjacency matrix (undirected)
    for e_i in range(n_e):
        nodes = incidence_list[e_i]  # nodes in this edge
        EDVWs = parameter_list[e_i]  # EDVWs associated with this edge
        esize = len(nodes)           # the edge cardinality
        for i in range(esize):
            vi = nodes[i]
            for j in range(i+1, esize):
                vj = nodes[j]
                A[vi,vj] = A[vi,vj] + edge_weight[e_i] * EDVWs[i] * EDVWs[j]
                A[vj,vi] = A[vi,vj]
                
    _s, _t = n_v, n_v+1
    for i in c0_labeled:
        A[_s, i] = inf
        A[i, _s] = inf
    for i in c1_labeled:
        A[_t, i] = inf
        A[i, _t] = inf

    return A, _s, _t

def star_expansion(incidence_list, parameter_list, edge_weight, n_v, n_e, c0_labeled, c1_labeled):

    g_n_v = n_v + n_e + 2
    A = np.zeros((g_n_v, g_n_v))
    for e_i in range(n_e):
        nodes = incidence_list[e_i]
        EDVWs = parameter_list[e_i]
        v_e = n_v + e_i  # introduce a vertex for this edge
        for v_i, v in enumerate(nodes):
            A[v, v_e] = edge_weight[e_i] * EDVWs[v_i]
            A[v_e, v] = A[v, v_e]

    _s = n_v + n_e
    _t = n_v + n_e + 1
    for i in c0_labeled:
        A[_s, i] = inf
        A[i, _s] = inf
    for i in c1_labeled:
        A[_t, i] = inf
        A[i, _t] = inf
        
    return A, _s, _t

def lawler_expansion(incidence_list, parameter_list, edge_weight, n_v, n_e, alpha, c0_labeled, c1_labeled):
    
    g_n_v = n_v + 2 * n_e + 2            # the graph size
    A = np.zeros((g_n_v, g_n_v))         # the adjacency matrix (digraph)
    for e_i in range(n_e):               # for each hyperedge
        nodes = incidence_list[e_i]      # nodes in this edge
        EDVWs = parameter_list[e_i]      # EDVWs associated with this edge
        EDVWs = EDVWs * edge_weight[e_i]
        e1 = n_v + e_i * 2               # introduce two auxiliary nodes
        e2 = n_v + e_i * 2 + 1
        A[e1, e2] = alpha * np.sum(EDVWs)
        for v_i, v in enumerate(nodes):
            A[v, e1] = EDVWs[v_i]
            A[e2, v] = EDVWs[v_i]
    
    _s = n_v + 2 * n_e      # the source 
    _t = n_v + 2 * n_e + 1  # the target
    for i in c0_labeled:
        A[_s, i] = inf
    for i in c1_labeled:
        A[i, _t] = inf
        
    return A, _s, _t

def all_or_nothing(incidence_list, parameter_list, edge_weight, n_v, n_e, c0_labeled, c1_labeled):
    
    g_n_v = n_v + 2 * n_e + 2            # the graph size
    A = np.zeros((g_n_v, g_n_v))         # the adjacency matrix (digraph)
    for e_i in range(n_e):               # for each hyperedge
        nodes = incidence_list[e_i]      # nodes in this edge
        EDVWs = parameter_list[e_i]      # EDVWs associated with this edge
        EDVWs = EDVWs * edge_weight[e_i]
        e1 = n_v + e_i * 2               # introduce two auxiliary nodes
        e2 = n_v + e_i * 2 + 1
        A[e1, e2] = np.sum(EDVWs)
        for v_i, v in enumerate(nodes):
            A[v, e1] = inf
            A[e2, v] = inf
    
    _s = n_v + 2 * n_e      # the source 
    _t = n_v + 2 * n_e + 1  # the target
    for i in c0_labeled:
        A[_s, i] = inf
    for i in c1_labeled:
        A[i, _t] = inf
        
    return A, _s, _t

def build_hg(tfidf, para_p):
    
    n_e, _ = np.shape(tfidf)
    incidence_list = {}
    parameter_list = {}
    for e_i in range(n_e):
        incidence_list[e_i] = np.where(tfidf[e_i] > 0)[0]
        parameter_list[e_i] = tfidf[e_i, incidence_list[e_i]] ** para_p
        
#         edvws = parameter_list[e_i]
#         assert np.sum(edvws)*0.0003 < np.min(edvws)
    
    return incidence_list, parameter_list
       
def classification(incidence_list, parameter_list, n_v, n_e, c0_labeled, c1_labeled, expansion, alpha=None):
    
    edge_weight = np.ones(n_e)
            
    if expansion == 'c':
        A, _s, _t = clique_expansion(incidence_list, parameter_list, edge_weight, n_v, n_e, c0_labeled, c1_labeled)
        flowG = nx.from_numpy_matrix(A)
        assert nx.is_connected(flowG)
    elif expansion == 's':
        A, _s, _t = star_expansion(incidence_list, parameter_list, edge_weight, n_v, n_e, c0_labeled, c1_labeled)
        flowG = nx.from_numpy_matrix(A)
        assert nx.is_connected(flowG)
    elif expansion == 'l':
        A, _s, _t = lawler_expansion(incidence_list, parameter_list, edge_weight, n_v, n_e, alpha, c0_labeled, c1_labeled)
        flowG = nx.from_numpy_matrix(A, create_using=nx.DiGraph)
        assert nx.is_weakly_connected(flowG)
    else:  # 'h'
        A, _s, _t = all_or_nothing(incidence_list, parameter_list, edge_weight, n_v, n_e, c0_labeled, c1_labeled)
        flowG = nx.from_numpy_matrix(A, create_using=nx.DiGraph)
        assert nx.is_weakly_connected(flowG)
         
    cut_value, partition = nx.minimum_cut(flowG, _s, _t, capacity='weight', flow_func=None)
        
    pred_classes = np.zeros(n_v)
    for i in partition[0]:
        if i < n_v:
            pred_classes[i] = 0
    for i in partition[1]:
        if i < n_v:
            pred_classes[i] = 1
    
    return pred_classes

def train_test_split(c0_index, c1_index, c0_size, c1_size, n_v, train_ratio):
    
    c0_labeled = random.sample(c0_index, int(c0_size * train_ratio))
    c1_labeled = random.sample(c1_index, int(c1_size * train_ratio))
   
    unlabeled = [True] * n_v
    for i in c0_labeled:
        unlabeled[i] = False
    for i in c1_labeled:
        unlabeled[i] = False
        
    return c0_labeled, c1_labeled, unlabeled
    
def train_val_split(c0_labeled_split, c1_labeled_split, n_cval, i_cval):
    
    c0_train, c1_train = [], []
    for i in range(n_cval):
        if not i == i_cval:
            c0_train = c0_train + c0_labeled_split[i]
            c1_train = c1_train + c1_labeled_split[i]
    
    c0_val = c0_labeled_split[i_cval]
    c1_val = c1_labeled_split[i_cval]
    total_val = c0_val + c1_val
   
    return c0_train, c1_train, total_val

def most_common(lst):
    return max(set(lst), key=lst.count)
