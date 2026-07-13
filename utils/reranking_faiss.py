#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri, 25 May 2018 20:29:09


"""

"""
CVPR2017 paper:Zhong Z, Zheng L, Cao D, et al. Re-ranking Person Re-identification with k-reciprocal Encoding[J]. 2017.
url:http://openaccess.thecvf.com/content_cvpr_2017/papers/Zhong_Re-Ranking_Person_Re-Identification_CVPR_2017_paper.pdf
Matlab version: https://github.com/zhunzhong07/person-re-ranking
"""

"""
API

probFea: all feature vectors of the query set (torch tensor)
probFea: all feature vectors of the gallery set (torch tensor)
k1,k2,lambda: parameters, the original paper is (k1=20,k2=6,lambda=0.3)
MemorySave: set to 'True' when using MemorySave mode
Minibatch: avaliable when 'MemorySave' is 'True'
"""

import numpy as np
import torch
import time

import faiss


def _faiss_knn_search(x, y=None, k=None, try_gpu=True):
    """
    x: np.ndarray [num_queries, dim] float32
    y: np.ndarray [num_vectors, dim] float32 (if None, y = x)
    k: top-k to search (if None, search all vectors in y)
    return: (D, I) distances and indices
    """
    if y is None:
        y = x
    dim = x.shape[1]
    if k is None:
        k = y.shape[0]

    index = faiss.IndexFlatL2(dim)
    use_gpu = False
    if try_gpu and hasattr(faiss, 'StandardGpuResources'):
        try:
            res = faiss.StandardGpuResources()
            index = faiss.index_cpu_to_gpu(res, 0, index)
            use_gpu = True
        except Exception:
            use_gpu = False

    index.add(y)
    D, I = index.search(x, k)

    # Explicitly free GPU resources if used
    if use_gpu:
        try:
            del res
        except Exception:
            pass

    return D, I


def _faiss_l2_matrix(x, y=None, try_gpu=True):
    """
    Build full pairwise L2 distance matrix using FAISS search.
    x: np.ndarray [n, d] float32
    y: np.ndarray [m, d] float32 or None (None => y=x)
    return: distances np.ndarray [n, m] float32
    """
    if y is None:
        y = x
    n = x.shape[0]
    m = y.shape[0]
    D, I = _faiss_knn_search(x, y=y, k=m, try_gpu=try_gpu)
    dist = np.empty((n, m), dtype=np.float32)
    row_idx = np.arange(n)[:, None]
    dist[row_idx, I] = D
    return dist


def re_ranking(probFea, galFea, k1, k2, lambda_value, local_distmat=None, only_local=False):
    # if feature vector is numpy, you should use 'torch.tensor' transform it to tensor
    query_num = probFea.size(0)
    all_num = query_num + galFea.size(0)
    if only_local:
        original_dist = local_distmat
    else:
        feat = torch.cat([probFea, galFea])
        feat_np = feat.detach().cpu().numpy().astype('float32')
        original_dist = _faiss_l2_matrix(
            feat_np,
            y=None,
            try_gpu=torch.cuda.is_available()
        )
        del feat
        if local_distmat is not None:
            local_np = local_distmat if isinstance(local_distmat, np.ndarray) else local_distmat.cpu().numpy()
            original_dist = original_dist + local_np
    gallery_num = original_dist.shape[0]
    original_dist = np.transpose(original_dist / np.max(original_dist, axis=0))
    V = np.zeros_like(original_dist).astype(np.float16)
    initial_rank = np.argsort(original_dist).astype(np.int32)

    # print('starting re_ranking')
    for i in range(all_num):
        # k-reciprocal neighbors
        forward_k_neigh_index = initial_rank[i, :k1 + 1]
        backward_k_neigh_index = initial_rank[forward_k_neigh_index, :k1 + 1]
        fi = np.where(backward_k_neigh_index == i)[0]
        k_reciprocal_index = forward_k_neigh_index[fi]
        k_reciprocal_expansion_index = k_reciprocal_index
        for j in range(len(k_reciprocal_index)):
            candidate = k_reciprocal_index[j]
            candidate_forward_k_neigh_index = initial_rank[candidate, :int(np.around(k1 / 2)) + 1]
            candidate_backward_k_neigh_index = initial_rank[candidate_forward_k_neigh_index,
                                               :int(np.around(k1 / 2)) + 1]
            fi_candidate = np.where(candidate_backward_k_neigh_index == candidate)[0]
            candidate_k_reciprocal_index = candidate_forward_k_neigh_index[fi_candidate]
            if len(np.intersect1d(candidate_k_reciprocal_index, k_reciprocal_index)) > 2 / 3 * len(
                    candidate_k_reciprocal_index):
                k_reciprocal_expansion_index = np.append(k_reciprocal_expansion_index, candidate_k_reciprocal_index)

        k_reciprocal_expansion_index = np.unique(k_reciprocal_expansion_index)
        weight = np.exp(-original_dist[i, k_reciprocal_expansion_index])
        V[i, k_reciprocal_expansion_index] = weight / np.sum(weight)
    original_dist = original_dist[:query_num, ]
    if k2 != 1:
        V_qe = np.zeros_like(V, dtype=np.float16)
        for i in range(all_num):
            V_qe[i, :] = np.mean(V[initial_rank[i, :k2], :], axis=0)
        V = V_qe
        del V_qe
    del initial_rank
    invIndex = []
    for i in range(gallery_num):
        invIndex.append(np.where(V[:, i] != 0)[0])

    jaccard_dist = np.zeros_like(original_dist, dtype=np.float16)

    for i in range(query_num):
        temp_min = np.zeros(shape=[1, gallery_num], dtype=np.float16)
        indNonZero = np.where(V[i, :] != 0)[0]
        indImages = [invIndex[ind] for ind in indNonZero]
        for j in range(len(indNonZero)):
            temp_min[0, indImages[j]] = temp_min[0, indImages[j]] + np.minimum(V[i, indNonZero[j]],
                                                                               V[indImages[j], indNonZero[j]])
        jaccard_dist[i] = 1 - temp_min / (2 - temp_min)

    final_dist = jaccard_dist * (1 - lambda_value) + original_dist * lambda_value
    del original_dist
    del V
    del jaccard_dist  
    final_dist = final_dist[:query_num, query_num:]
    return final_dist

def k_reciprocal_neigh(initial_rank, i, k1):
    forward_k_neigh_index = initial_rank[i,:k1+1]
    backward_k_neigh_index = initial_rank[forward_k_neigh_index,:k1+1]
    fi = torch.nonzero(backward_k_neigh_index==i)[:,0]
    return forward_k_neigh_index[fi]

def compute_jaccard_dist(target_features, k1=20, k2=6, print_flag=True,          # 先对得到的特征做了一个重排序，然后再计算杰卡德距离
                        lambda_value=0, source_features=None, use_gpu=False):   
    end = time.time()
    N = target_features.size(0)
    if use_gpu:
        # accelerate matrix distance computing
        target_features = target_features.cuda()
        if (source_features is not None):
            source_features = source_features.cuda()

    if ((lambda_value>0) and (source_features is not None)):
        # 使用 FAISS 计算 target 与 source 的两两 L2 距离
        src_np = source_features.detach().cpu().numpy().astype('float32')
        tar_np = target_features.detach().cpu().numpy().astype('float32')
        sour_tar_dist_np = _faiss_l2_matrix(tar_np, y=src_np, try_gpu=torch.cuda.is_available() and use_gpu)
        sour_tar_dist_np = 1 - np.exp(-sour_tar_dist_np)
        source_dist_vec_np = sour_tar_dist_np.min(axis=1)
        del sour_tar_dist_np
        source_dist_vec = torch.from_numpy(source_dist_vec_np)
        source_dist_vec /= source_dist_vec.max()   # 对 1xN 矩阵每个元素除以最大值，目的？
        source_dist = torch.zeros(N, N)
        for i in range(N):
            source_dist[i, :] = source_dist_vec + source_dist_vec[i]
        del source_dist_vec


    if print_flag:
        pass
        # print('Computing original distance...')      # 计算目标域特征之间的欧式距离，original_dist的维度是 n x n , n为图片数量

    # 使用 FAISS 生成全对距离矩阵与初始排序
    tar_np = target_features.detach().cpu().numpy().astype('float32')
    original_dist_np = _faiss_l2_matrix(tar_np, y=None, try_gpu=torch.cuda.is_available() and use_gpu)
    # 列归一化并转置，与原逻辑保持一致
    original_dist_np = (original_dist_np / original_dist_np.max(axis=0, keepdims=True)).T
    initial_rank_np = np.argsort(original_dist_np, axis=-1)
    original_dist = torch.from_numpy(original_dist_np)
    initial_rank = torch.from_numpy(initial_rank_np)
    #print(initial_rank)  # 

    original_dist = original_dist.cpu()
    initial_rank = initial_rank.cpu()
    all_num = gallery_num = original_dist.size(0)         

    del target_features
    if (source_features is not None):
        del source_features

    if print_flag:
        pass
        # print('Computing Jaccard distance...')

    nn_k1 = []
    nn_k1_half = []
    for i in range(all_num):
        nn_k1.append(k_reciprocal_neigh(initial_rank, i, k1))
        nn_k1_half.append(k_reciprocal_neigh(initial_rank, i, int(np.around(k1/2))))

    V = torch.zeros(all_num, all_num)
    for i in range(all_num):
        k_reciprocal_index = nn_k1[i]
        k_reciprocal_expansion_index = k_reciprocal_index
        for candidate in k_reciprocal_index:
            candidate_k_reciprocal_index = nn_k1_half[candidate]
            if (len(np.intersect1d(candidate_k_reciprocal_index,k_reciprocal_index)) > 2/3*len(candidate_k_reciprocal_index)):
                k_reciprocal_expansion_index = torch.cat((k_reciprocal_expansion_index,candidate_k_reciprocal_index))

        k_reciprocal_expansion_index = torch.unique(k_reciprocal_expansion_index)  ## element-wise unique
        weight = torch.exp(-original_dist[i,k_reciprocal_expansion_index])
        V[i,k_reciprocal_expansion_index] = weight/torch.sum(weight)

    if k2 != 1:
        k2_rank = initial_rank[:,:k2].clone().view(-1)
        V_qe = V[k2_rank]
        V_qe = V_qe.view(initial_rank.size(0),k2,-1).sum(1)
        V_qe /= k2
        V = V_qe
        del V_qe
    del initial_rank

    invIndex = []
    for i in range(gallery_num):
        invIndex.append(torch.nonzero(V[:,i])[:,0])  #len(invIndex)=all_num

    jaccard_dist = torch.zeros_like(original_dist)
    for i in range(all_num):
        temp_min = torch.zeros(1,gallery_num)
        indNonZero = torch.nonzero(V[i,:])[:,0]
        indImages = []
        indImages = [invIndex[ind] for ind in indNonZero]
        for j in range(len(indNonZero)):
            temp_min[0,indImages[j]] = temp_min[0,indImages[j]]+ torch.min(V[i,indNonZero[j]],V[indImages[j],indNonZero[j]])
        jaccard_dist[i] = 1-temp_min/(2-temp_min)
    del invIndex

    del V

    pos_bool = (jaccard_dist < 0)
    jaccard_dist[pos_bool] = 0.0
    if print_flag:
        print ("Time cost: {}".format(time.time()-end))
    
    if (lambda_value>0):
        return jaccard_dist*(1-lambda_value) + source_dist*lambda_value
    else:
        return jaccard_dist