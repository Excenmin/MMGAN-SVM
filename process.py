# -*- coding: utf-8 -*-


import numpy as np
import scipy.sparse as sp
import scipy.spatial.distance as dist
from utils import row_normalize

def loaddata(path):
    mat = np.loadtxt(path, dtype=np.float32,  delimiter=' ')
    return mat


def get_mateneigh_adj(mat):
    k = 1
    n = mat.shape[0]
    mateneigh_adj = np.zeros([n, n],dtype=float)
    for i in range(n):
        for j in range(i+1, n):
            if np.dot(mat[i], mat[j]) > k:

                mateneigh_adj[i][j] = 1
                mateneigh_adj[j][i] = 1

    return mateneigh_adj


def divide_known_unknown_associations(A, exception=None, special=None):
    known = []
    unknown = []
    if special != None:
        for i in range(A.shape[0]):
            if A[i][special] == 1:
                known.append([i,special,1])
            else:
                unknown.append([i,special,0])
    else:
        for i in range(A.shape[0]):
            for j in range(A.shape[1]):
                if j == exception: pass
                if A[i][j] == 1:
                    known.append([i,j,1])
                else:
                    unknown.append([i,j,0])

    return np.array(known), np.array(unknown)


def constructHNet(mask_adj):

    mp = loaddata('./data/MP_mat.txt')
    dp = loaddata('./data/DP_mat.txt')
    mm = loaddata('./data/miRNA_FuncSM_Matrix.txt')
    mm_gip = loaddata('./data/GIPM_mat.txt')
    dd = loaddata('./data/Disease_SemanticSM_Matrix.txt')
    dd_gip = loaddata('./data/GIPD_mat.txt')


    mm_d = get_mateneigh_adj(mask_adj) + np.eye(mm.shape[0])
    mm_p = get_mateneigh_adj(mp) + np.eye(mm.shape[0])
    dd_m = get_mateneigh_adj(mask_adj.T) + np.eye(dd.shape[0])
    dd_p = get_mateneigh_adj(dp) + np.eye(dd.shape[0])


    mm = mm * 0.5 + mm_gip * 0.5
    dd = dd * 0.5 + dd_gip * 0.5
    # mm = row_normalize(mm.T, substract_self_loop=False).T
    # dd = row_normalize(dd.T, substract_self_loop=False).T
    # mm_gip = row_normalize(mm_gip.T, substract_self_loop=False).T
    # dd_gip = row_normalize(dd_gip.T, substract_self_loop=False).T

    # mm = mm * 0.5 + mm_gip * 0.5
    # dd = dd * 0.5 + dd_gip * 0.5
    # mm_d = row_normalize(mm_d, substract_self_loop=False)
    # mm_p = row_normalize(mm_p, substract_self_loop=False)
    # dd_m = row_normalize(dd_m, substract_self_loop=False)
    # dd_p = row_normalize(dd_p, substract_self_loop=False)

    tmp1 = np.concatenate((mm, mask_adj.T), axis=0)
    tmp2 = np.concatenate((mask_adj, dd), axis=0)
    MDA1 = np.concatenate((tmp1, tmp2), axis=1)


    tmp1 = np.concatenate((mm_d, mask_adj.T), axis=0)
    tmp2 = np.concatenate((mask_adj, dd_m), axis=0)
    MDA2 = np.concatenate((tmp1, tmp2), axis=1)

    tmp1 = np.concatenate((mm_p, mask_adj.T), axis=0)
    tmp2 = np.concatenate((mask_adj, dd_p), axis=0)
    MDA3 = np.concatenate((tmp1, tmp2), axis=1)

    mean_adj = (MDA1 + MDA2 + MDA3) / 3


    # MDA1 = sp.csr_matrix(MDA1)
    # MDA2 = sp.csr_matrix(MDA2)
    # MDA3 = sp.csr_matrix(MDA3)

    return MDA1, MDA2, MDA3, mean_adj

