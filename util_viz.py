import math
from math import sqrt, log, pi
import numpy as np
import pandas as pd
from Bio import Phylo
from Bio.Phylo import NewickIO

def projective_2coords(tree):
    '''
    function which takes a 3-leaf tree and returns the
    2-dimensional coordinates for external projective space
    '''
    ## Vertices (position-vectors) for 3-branch projective external space (Shell)
    N3 = np.array([0,0,1])
    P3 = np.array([1,0,0])
    S3 = np.array([0,1,0])

    weights = pd.DataFrame(np.zeros((1,3)), columns=['____Normal','___Primary','_Secondary'])
    named_weights = [(clade.name, clade.branch_length) for clade in tree.get_terminals()]
    for name,blength in named_weights:
        if name in weights.columns:
            weights[name][0] = blength
    weights = weights / np.sqrt(np.sum(np.array(weights)**2))
    coords = weights['___Primary'][0]*P3 + weights['_Secondary'][0]*S3 + weights['____Normal'][0]*N3
    return coords

def projective_3coords(tree):
    '''
    function which takes a 4-leaf tree and returns the
    3-dimensional coordinates for external projective space
    '''
    ## Vertices (position vectors) for 4-branch projective external space

    ## N4: (-0.29, 0.5, -0.20)
    N4 = np.array([-1.0/(2*sqrt(3)), 1.0/2, -1.0/(2*sqrt(6))])

    ## P4: (-0.29, -0.5, -0.20)
    P4 = np.array([-1.0/(2*sqrt(3)), -1.0/2, -1.0/(2*sqrt(6))])

    ## S4: (0.29, 0, -0.20)
    S4 = np.array([1.0/sqrt(3), 0, -1.0/(2*sqrt(6))])

    ## T4: (0, 0, 0.61)
    T4 = np.array([0, 0, sqrt(2.0/3) - 1.0/(2*sqrt(6))])

    noise = 1 - np.abs(np.random.random()/20)
    internal_node = tree.get_nonterminals()[-1]
    
    weights = pd.DataFrame(np.zeros((1,5)), columns=['____Normal','___Primary','_Secondary','__Tertiary','Internal'])
    if internal_node.branch_length > 0:
        weights['Internal'][0] = internal_node.branch_length
    else:
        weights['Internal'][0] = tree.get_nonterminals()[-2].branch_length
    named_weights = [(clade.name, clade.branch_length) for clade in tree.get_terminals()]
    for name,blength in named_weights:
        if name in weights.columns:
            weights[name][0] = blength
    weights = weights / np.sum(np.array(weights))
    coords = noise * (weights['___Primary'][0]*P4 + weights['_Secondary'][0]*S4 + \
                      weights['__Tertiary'][0]*T4 + weights['____Normal'][0]*N4)
    return coords

def projective_4coords_shadowLast(tree):
    '''
    function which takes a 5-leaf tree and returns the
    3-dimensional coordinates for the shadow of the external
    projective space viewed thru the Last sample
    '''
    ## Vertices (position vectors) for 4-branch projective external space

    ## N4: (-0.29, 0.5, -0.20)
    N4 = np.array([-1.0/(2*sqrt(3)), 1.0/2, -1.0/(2*sqrt(6))])

    ## P4: (-0.29, -0.5, -0.20)
    P4 = np.array([-1.0/(2*sqrt(3)), -1.0/2, -1.0/(2*sqrt(6))])

    ## S4: (0.29, 0, -0.20)
    S4 = np.array([1.0/sqrt(3), 0, -1.0/(2*sqrt(6))])

    ## T4: (0, 0, 0.61)
    T4 = np.array([0, 0, sqrt(2.0/3) - 1.0/(2*sqrt(6))])

    weights = pd.DataFrame(np.zeros((1,5)), columns=['____Normal','___Primary','_Secondary','__Tertiary', 'Quaternary'])
    named_weights = [(clade.name, clade.branch_length) for clade in tree.get_terminals()]
    for name,blength in named_weights:
        if name in weights.columns:
            weights[name][0] = blength
    weights = weights / np.sum(np.array(weights))
    noise = 1 - np.abs(np.random.random()/20)
    coords = noise * (weights['___Primary'][0]*P4 + weights['_Secondary'][0]*S4 + \
                      weights['__Tertiary'][0]*T4 + weights['____Normal'][0]*N4)
    return coords

def projective_4coords_shadowFirst(tree):
    '''
    function which takes a 5-leaf tree and returns the
    3-dimensional coordinates for the shadow of the external
    projective space viewed thru the First sample
    '''
    ## Vertices (position vectors) for 4-branch projective external space

    ## N4: (-0.29, 0.5, -0.20)
    N4 = np.array([-1.0/(2*sqrt(3)), 1.0/2, -1.0/(2*sqrt(6))])
    Q4_prime = N4

    ## P4: (-0.29, -0.5, -0.20)
    P4 = np.array([-1.0/(2*sqrt(3)), -1.0/2, -1.0/(2*sqrt(6))])
    T4_prime = P4

    ## S4: (0.29, 0, -0.20)
    S4 = np.array([1.0/sqrt(3), 0, -1.0/(2*sqrt(6))])
    S4_prime = S4

    ## T4: (0, 0, 0.61)
    T4 = np.array([0, 0, sqrt(2.0/3) - 1.0/(2*sqrt(6))])
    P4_prime = T4

    weights = pd.DataFrame(np.zeros((1,5)), columns=['____Normal','___Primary','_Secondary','__Tertiary', 'Quaternary'])
    named_weights = [(clade.name, clade.branch_length) for clade in tree.get_terminals()]
    for name,blength in named_weights:
        if name in weights.columns:
            weights[name][0] = blength
    weights = weights / np.sum(np.array(weights))
    noise = 1 - np.abs(np.random.random()/20)
    coords = noise * (weights['___Primary'][0]*P4_prime + weights['_Secondary'][0]*S4_prime + \
                      weights['__Tertiary'][0]*T4_prime + weights['Quaternary'][0]*Q4_prime)
    return coords

def recover_topology_quad(tree):
    '''
    function which takes a 4-leaf tree and returns which
    of the 3 possible topologies it displays, along with
    the length of the internal branch
    '''
    noise = 1 - np.abs(np.random.random()/5)
    internal_node = tree.get_nonterminals()[-1]
    
    weights = pd.DataFrame(np.zeros((1,5)), columns=['____Normal','___Primary','_Secondary','__Tertiary','Internal'])
    if internal_node.branch_length > 0:
        weights['Internal'][0] = internal_node.branch_length
    else:
        weights['Internal'][0] = tree.get_nonterminals()[-2].branch_length
    named_weights = [(clade.name, clade.branch_length) for clade in tree.get_terminals()]
    for name,blength in named_weights:
        if name in weights.columns:
            weights[name][0] = blength
    weights = weights / np.sum(np.array(weights))
    
    blen = noise * weights['Internal'][0]
    topology = set([leaf.name for leaf in internal_node.clades])
    
    # degenerate case of all internal branches collapsing to zero
    if blen == 0:
        print topology
        return 0, 0
    
    if topology == set(['____Normal','___Primary']) or topology == set(['_Secondary','__Tertiary']):
        return 1, blen
    elif topology == set(['____Normal','_Secondary']) or topology == set(['___Primary','__Tertiary']):
        return 2, blen
    elif topology == set(['____Normal','__Tertiary']) or topology == set(['___Primary','_Secondary']):
        return 3, blen
    
    # error, for some reason we reached this point in the function
    else:
        print topology
        return -1, 0

def recover_topology_quint(tree):
    '''
    function which takes a 5-leaf tree and returns which
    of the 15 possible topologies it displays, along with
    the lengths of the two internal branches
    '''
    couplet_NP = set(['____Normal','___Primary'])
    couplet_NS = set(['____Normal','_Secondary'])
    couplet_NT = set(['____Normal','__Tertiary'])
    couplet_NQ = set(['____Normal','Quaternary'])
    couplet_PS = set(['___Primary','_Secondary'])
    couplet_PT = set(['___Primary','__Tertiary'])
    couplet_PQ = set(['___Primary','Quaternary'])
    couplet_ST = set(['_Secondary','__Tertiary'])
    couplet_SQ = set(['_Secondary','Quaternary'])
    couplet_TQ = set(['__Tertiary','Quaternary'])
    
    internal_nodes = tree.get_nonterminals()
    internal_blen_L = internal_nodes[1].branch_length
    internal_blen_R = internal_nodes[2].branch_length
    external_nodes = tree.get_terminals()
    couplet_L = set([cl.name for cl in internal_nodes[0].clades if cl in external_nodes])
    couplet_R = set([cl.name for cl in internal_nodes[2].clades if cl in external_nodes])
    singleton = set([cl.name for cl in internal_nodes[1].clades if cl in external_nodes])
    if '____Normal' in couplet_L:
        blen_1 = internal_blen_L
        blen_2 = internal_blen_R
    elif '__Normal' in couplet_R:
        blen_1 = internal_blen_R
        blen_2 = internal_blen_L
    else:
        if '___Primary' in couplet_L:
            blen_1 = internal_blen_L
            blen_2 = internal_blen_R
        else:
            blen_1 = internal_blen_R
            blen_2 = internal_blen_L
    
    #noise = 1 - np.abs(np.random.random()/10)
    noise = 1
    blen_1 = noise * blen_1
    blen_2 = noise * blen_2
    
    # degenerate case of all internal branches collapsing to zero
    if blen_1 == 0 and blen_2 == 0:
        return 0, 0, 0
    
    if singleton == set(['____Normal']) and (couplet_L == couplet_PS or couplet_L == couplet_TQ):
        return 1, blen_1, blen_2
    elif singleton == set(['Quaternary']) and (couplet_L == couplet_NT or couplet_L == couplet_PS):
        return 2, blen_1, blen_2
    elif singleton == set(['___Primary']) and (couplet_L == couplet_NT or couplet_L == couplet_SQ):
        return 3, blen_1, blen_2
    elif singleton == set(['__Tertiary']) and (couplet_L == couplet_NP or couplet_L == couplet_SQ):
        return 4, blen_1, blen_2
    elif singleton == set(['_Secondary']) and (couplet_L == couplet_NP or couplet_L == couplet_TQ):
        return 5, blen_1, blen_2
    elif singleton == set(['___Primary']) and (couplet_L == couplet_NS or couplet_L == couplet_TQ):
        return 6, blen_1, blen_2
    elif singleton == set(['__Tertiary']) and (couplet_L == couplet_NQ or couplet_L == couplet_PS):
        return 7, blen_1, blen_2
    elif singleton == set(['_Secondary']) and (couplet_L == couplet_NT or couplet_L == couplet_PQ):
        return 8, blen_1, blen_2
    elif singleton == set(['____Normal']) and (couplet_L == couplet_PT or couplet_L == couplet_SQ):
        return 9, blen_1, blen_2
    elif singleton == set(['Quaternary']) and (couplet_L == couplet_NP or couplet_L == couplet_ST):
        return 10, blen_1, blen_2
    elif singleton == set(['__Tertiary']) and (couplet_L == couplet_NS or couplet_L == couplet_PQ):
        return 11, blen_1, blen_2
    elif singleton == set(['Quaternary']) and (couplet_L == couplet_NS or couplet_L == couplet_PT):
        return 12, blen_1, blen_2
    elif singleton == set(['_Secondary']) and (couplet_L == couplet_NQ or couplet_L == couplet_PT):
        return 13, blen_1, blen_2
    elif singleton == set(['___Primary']) and (couplet_L == couplet_NQ or couplet_L == couplet_ST):
        return 14, blen_1, blen_2
    elif singleton == set(['____Normal']) and (couplet_L == couplet_PQ or couplet_L == couplet_ST):
        return 15, blen_1, blen_2
    
    # error, for some reason we reached this point in the function
    return -1, 0, 0