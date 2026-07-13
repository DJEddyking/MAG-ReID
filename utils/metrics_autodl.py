from prettytable import PrettyTable
import torch
import numpy as np
import os
import torch.nn.functional as F
import logging
from model.clip_model import InputImageType
from utils.reranking import re_ranking


def rank(similarity, q_pids, g_pids, max_rank=10, get_mAP=True):
    if get_mAP:
        indices = torch.argsort(similarity, dim=1, descending=True)
    else:
        # acclerate sort with topk
        _, indices = torch.topk(
            similarity, k=max_rank, dim=1, largest=True, sorted=True
        )  # q * topk
    pred_labels = g_pids[indices]  # q * k
    matches = pred_labels.eq(q_pids.view(-1, 1))  # q * k

    all_cmc = matches[:, :max_rank].cumsum(1) # cumulative sum
    all_cmc[all_cmc > 1] = 1
    all_cmc = all_cmc.float().mean(0) * 100
    # all_cmc = all_cmc[topk - 1]

    if not get_mAP:
        return all_cmc, indices

    num_rel = matches.sum(1)  # q
    tmp_cmc = matches.cumsum(1)  # q * k
    tmp_cmc = [tmp_cmc[:, i] / (i + 1.0) for i in range(tmp_cmc.shape[1])]
    tmp_cmc = torch.stack(tmp_cmc, 1) * matches
    AP = tmp_cmc.sum(1) / num_rel  # q
    mAP = AP.mean() * 100
    return all_cmc, mAP, indices


def eval_func(distmat, q_pids, g_pids, q_camids, g_camids, set=0, max_rank=50):
    """Evaluation with market1501 metric
        Key: for each query identity, its gallery images from the same camera view are discarded.
        """
    num_q, num_g = distmat.shape
    if num_g < max_rank:
        max_rank = num_g
        print("Note: number of gallery samples is quite small, got {}".format(num_g))
    indices = np.argsort(distmat, axis=1)
    matches = (g_pids[indices] == q_pids[:, np.newaxis]).astype(np.int32)

    # compute cmc curve for each query
    all_cmc = []
    all_AP = []
    all_INP = []
    num_valid_q = 0.  # number of valid query
    for q_idx in range(num_q):
        # get query pid and camid
        q_pid = q_pids[q_idx]
        q_camid = q_camids[q_idx]

        # remove gallery samples that have the same pid and camid with query
        if set == 2:
            order = indices[q_idx]
            remove = (g_pids[order] == q_pid) & (g_camids[order] == q_camid)
            keep = np.invert(remove)

            # compute cmc curve
            # binary vector, positions with value 1 are correct matches
            orig_cmc = matches[q_idx][keep]
        else:
            orig_cmc = matches[q_idx]

        if not np.any(orig_cmc):
            # this condition is true when query identity does not appear in gallery
            continue


        cmc = orig_cmc.cumsum()

        pos_idx = np.where(orig_cmc == 1)
        max_pos_idx = np.max(pos_idx)
        inp = cmc[max_pos_idx]/ (max_pos_idx + 1.0)
        all_INP.append(inp)

        cmc[cmc > 1] = 1

        all_cmc.append(cmc[:max_rank])
        num_valid_q += 1.

        # compute average precision
        # reference: https://en.wikipedia.org/wiki/Evaluation_measures_(information_retrieval)#Average_precision
        num_rel = orig_cmc.sum()
        tmp_cmc = orig_cmc.cumsum()
        tmp_cmc = [x / (i + 1.) for i, x in enumerate(tmp_cmc)]
        tmp_cmc = np.asarray(tmp_cmc) * orig_cmc
        AP = tmp_cmc.sum() / num_rel
        all_AP.append(AP)

    assert num_valid_q > 0, "Error: all query identities do not appear in gallery"

    all_cmc = np.asarray(all_cmc).astype(np.float32)
    all_cmc = all_cmc.sum(0) / num_valid_q
    mAP = np.mean(all_AP)
    mINP = np.mean(all_INP)

    return all_cmc* 100, mAP* 100, mINP* 100

class Evaluator():
    def __init__(self, args, val_img_loader, val_text_loader, val_sketch_loader, val_color_pencil_loader, val_nir_loader,\
            val_text_sk_loader, val_text_cp_loader, val_text_nir_loader, val_sk_cp_loader, val_sk_nir_loader, val_cp_nir_loader,\
            val_text_cp_sk_loader, val_text_cp_nir_loader, val_text_sk_nir_loader, val_cp_sk_nir_loader,\
            val_text_cp_sk_nir_loader, test_setting=0):
        self.img_loader = val_img_loader # gallery

        self.text_loader = val_text_loader # query1
        self.sketch_loader = val_sketch_loader # query2
        self.cp_loader = val_color_pencil_loader # query3
        self.nir_loader = val_nir_loader # query4
        self.text_sk_loader = val_text_sk_loader # query5
        self.text_cp_loader = val_text_cp_loader # query6
        self.text_nir_loader = val_text_nir_loader # query7

        self.sk_cp_loader = val_sk_cp_loader # query8
        self.sk_nir_loader = val_sk_nir_loader # query9
        self.cp_nir_loader = val_cp_nir_loader # query10
        self.text_cp_sk_loader = val_text_cp_sk_loader # query11
        self.text_cp_nir_loader = val_text_cp_nir_loader # query12
        self.text_sk_nir_loader = val_text_sk_nir_loader # query13
        self.cp_sk_nir_loader = val_cp_sk_nir_loader # query14
        self.text_cp_sk_nir_loader = val_text_cp_sk_nir_loader #query 15

        self.args = args

        self.test_setting = test_setting
        self.logger = logging.getLogger("CLIP2ReID.eval")

        self.all_query_ids_list = []
        self.all_gallery_ids_list = []

    def _compute_embedding(self, model, mode='train_val'):
        """测试过程: 可以尝试返回单模态特征, 最后检索用单模态 + fusion 模态,  共同计算检索结果, 再投票一下; 
        只用到了 cls , eos  token, 没用融合特征啊 !!!! ---> 效果似乎并不好
        """
        
        model = model.eval()
        device = next(model.parameters()).device
        #gallery
        gids, gfeats, gimage_ids = self._onemodal(self.img_loader, device, model, InputImageType.rgb)
        # gfeats = F.normalize(gfeats, p=2, dim=1) # image features

        # ----------------单模态----------------
        #sk,cp,nir
        qids_sketch, qfeats_sketch, qimage_ids_sketch = self._onemodal(self.sketch_loader, device, model, InputImageType.sketch)
        qids_cp, qfeats_cp, qimage_ids_cp = self._onemodal(self.cp_loader, device, model, InputImageType.color_pencil)
        qids_nir, qfeats_nir, qimage_ids_nir = self._onemodal(self.nir_loader, device, model, InputImageType.nir)
        qids_text, qfeats_text, qtext_ids = self._onemodal_t(self.text_loader, device, model)

        # onemodal_rank1, onekodal_rank5, onemodal_rank10, onemodal_map,onemodal_minp = 0,0,0,0,0
        res1=self._compute_sim(qfeats_text, gfeats, qids_text, gids, qtext_ids, gimage_ids, mode)
                            #    query_feats_tse=qfeats_text_tse, gallery_feats_tse=gfeats_tse)
        res2=self._compute_sim(qfeats_sketch, gfeats, qids_sketch, gids, qimage_ids_sketch, gimage_ids, mode)
                            #    query_feats_tse=qfeats_sketch_tse, gallery_feats_tse=gfeats_tse)
        res3=self._compute_sim(qfeats_cp, gfeats, qids_cp, gids, qimage_ids_cp, gimage_ids, mode)
                            #    query_feats_tse=qfeats_cp_tse, gallery_feats_tse=gfeats_tse)
        res4=self._compute_sim(qfeats_nir, gfeats, qids_nir, gids, qimage_ids_nir, gimage_ids, mode)
                            #    query_feats_tse=qfeats_nir_tse, gallery_feats_tse=gfeats_tse)

        if mode == 'train_val':
            onemodal_rank1 = (res1[0] + res2[0] + res3[0] + res4[0]) / 4
            onemodal_rank5 = (res1[1] + res2[1] + res3[1] + res4[1]) / 4
            onemodal_rank10 = (res1[2] + res2[2] + res3[2] + res4[2]) / 4
            onemodal_map = (res1[3] + res2[3] + res3[3] + res4[3]) / 4
            onemodal_minp = (res1[4] + res2[4] + res3[4] + res4[4]) / 4

        # ----------------双模态----------------

        # text+sketch, tc, tn, sc, sn, cn
        qids_t_sk, qfeats_text_sk, qimage_ids_t_sk, qfeats_text_sk_sk, qfeats_text_sk_text = self._twomodal_it(
            self.text_sk_loader, device, model, InputImageType.sketch)
        qids_t_cp, qfeats_text_cp, qimage_ids_t_cp, qfeats_text_cp_cp, qfeats_text_cp_text = self._twomodal_it(
            self.text_cp_loader, device, model, InputImageType.color_pencil)
        qids_t_nir, qfeats_text_nir, qimage_ids_t_nir, qfeats_text_nir_nir, qfeats_text_nir_text = self._twomodal_it(
            self.text_nir_loader, device, model, InputImageType.nir)
        qids_sk_cp, qfeats_sk_cp, qimage_ids_sk_cp, qfeats_sk_cp_sk, qfeats_sk_cp_cp = self._twomodal_ii(
            self.sk_cp_loader, device, model, InputImageType.sketch, InputImageType.color_pencil)
        qids_cp_nir, qfeats_cp_nir, qimage_ids_cp_nir, qfeats_cp_nir_cp, qfeats_cp_nir_nir= self._twomodal_ii(
            self.cp_nir_loader, device, model, InputImageType.color_pencil, InputImageType.nir)
        qids_sk_nir, qfeats_sk_nir, qimage_ids_sk_nir, qfeats_sk_nir_sk, qfeats_sk_nir_nir= self._twomodal_ii(
            self.sk_nir_loader, device, model, InputImageType.sketch, InputImageType.nir)
        res5=self._compute_sim(
            qfeats_text_sk, gfeats, qids_t_sk, gids, qimage_ids_t_sk, gimage_ids, mode, qfeats_text_sk_sk, qfeats_text_sk_text)
            # query_feats_1_tse=qfeats_text_sk_sk_tse, query_feats_2_tse=qfeats_text_sk_text_tse, gallery_feats_tse=gfeats_tse,
            # query_feats_tse=qfeats_text_sk_tse)
        res6=self._compute_sim(
            qfeats_text_cp, gfeats, qids_t_cp, gids, qimage_ids_t_cp, gimage_ids, mode, qfeats_text_cp_cp, qfeats_text_cp_text)
            # query_feats_1_tse=qfeats_text_cp_cp_tse, query_feats_2_tse=qfeats_text_cp_text_tse, gallery_feats_tse=gfeats_tse,
            # query_feats_tse=qfeats_text_cp_tse)
        res7=self._compute_sim(
            qfeats_text_nir, gfeats, qids_t_nir, gids, qimage_ids_t_nir, gimage_ids, mode, qfeats_text_nir_nir, qfeats_text_nir_text)
            # query_feats_1_tse=qfeats_text_nir_nir_tse, query_feats_2_tse=qfeats_text_nir_text_tse, gallery_feats_tse=gfeats_tse,
            # query_feats_tse=qfeats_text_nir_tse)
        res8=self._compute_sim(
            qfeats_sk_cp, gfeats, qids_sk_cp, gids, qimage_ids_sk_cp, gimage_ids, mode, qfeats_sk_cp_sk, qfeats_sk_cp_cp)
            # query_feats_1_tse=qfeats_sk_cp_sk_tse, query_feats_2_tse=qfeats_sk_cp_cp_tse, gallery_feats_tse=gfeats_tse,
            # query_feats_tse=qfeats_sk_cp_tse)
        res9=self._compute_sim(
            qfeats_cp_nir, gfeats, qids_cp_nir, gids, qimage_ids_cp_nir, gimage_ids, mode, qfeats_cp_nir_cp, qfeats_cp_nir_nir)
            # query_feats_1_tse=qfeats_cp_nir_cp_tse, query_feats_2_tse=qfeats_cp_nir_nir_tse, gallery_feats_tse=gfeats_tse,
            # query_feats_tse=qfeats_cp_nir_tse)
        res10=self._compute_sim(
            qfeats_sk_nir, gfeats, qids_sk_nir, gids, qimage_ids_sk_nir, gimage_ids, mode, qfeats_sk_nir_sk, qfeats_sk_nir_nir)
            # query_feats_1_tse=qfeats_sk_nir_sk_tse, query_feats_2_tse=qfeats_sk_nir_nir_tse, gallery_feats_tse=gfeats_tse,
            # query_feats_tse=qfeats_sk_nir_tse)
        if mode == 'train_val':
            twomodal_rank1 = (res5[0] + res6[0] + res7[0] + res8[0] + res9[0] + res10[0]) / 6
            twomodal_rank5 = (res5[1] + res6[1] + res7[1] + res8[1] + res9[1] + res10[1]) / 6
            twomodal_rank10 = (res5[2] + res6[2] + res7[2] + res8[2] + res9[2] + res10[2]) / 6
            twomodal_map = (res5[3] + res6[3] + res7[3] + res8[3] + res9[3] + res10[3]) / 6
            twomodal_minp = (res5[4] + res6[4] + res7[4] + res8[4] + res9[4] + res10[4]) / 6


        # ----------------三模态----------------
        qids_t_cn, qfeats_text_cp_nir, qimage_ids_tcn, qfeats_text_cp_nir_cp, qfeats_text_cp_nir_nir, qfeats_text_cp_nir_text= self._threemodal_iit(
            self.text_cp_nir_loader, device, model, InputImageType.color_pencil, InputImageType.nir)
        qids_t_sn, qfeats_text_sk_nir, qimage_ids_tsn, qfeats_text_sk_nir_sk, qfeats_text_sk_nir_nir, qfeats_text_sk_nir_text= self._threemodal_iit(
            self.text_sk_nir_loader, device, model, InputImageType.sketch, InputImageType.nir)
        qids_t_cs, qfeats_text_cp_sk, qimage_ids_tcs, qfeats_text_cp_sk_cp, qfeats_text_cp_sk_sk, qfeats_text_cp_sk_text= self._threemodal_iit(
            self.text_cp_sk_loader, device, model, InputImageType.sketch, InputImageType.color_pencil)
        qids_csn, qfeats_cp_sk_nir, qimage_ids_csn, qfeats_cp_sk_nir_cp, qfeats_cp_sk_nir_sk, qfeats_cp_sk_nir_nir= self._threemodal_iii(
            self.cp_sk_nir_loader, device, model, InputImageType.sketch, InputImageType.color_pencil, InputImageType.nir)
        res11=self._compute_sim(
            qfeats_text_cp_nir, gfeats, qids_t_cn, gids, qimage_ids_tcn, gimage_ids, mode, 
            qfeats_text_cp_nir_cp, qfeats_text_cp_nir_nir, qfeats_text_cp_nir_text)
            # query_feats_1_tse=qfeats_text_cp_nir_cp_tse, query_feats_2_tse=qfeats_text_cp_nir_nir_tse, 
            # query_feats_3_tse=qfeats_text_cp_nir_text_tse, gallery_feats_tse=gfeats_tse,
            # query_feats_tse=qfeats_text_cp_nir_tse)
        res12=self._compute_sim(qfeats_text_sk_nir, gfeats, qids_t_sn, gids, qimage_ids_tsn, gimage_ids, mode,
            qfeats_text_sk_nir_sk, qfeats_text_sk_nir_nir, qfeats_text_sk_nir_text)
            # query_feats_1_tse=qfeats_text_sk_nir_sk_tse, query_feats_2_tse=qfeats_text_sk_nir_nir_tse, 
            # query_feats_3_tse=qfeats_text_sk_nir_text_tse, gallery_feats_tse=gfeats_tse,
            # query_feats_tse=qfeats_text_sk_nir_tse)
        res13=self._compute_sim(qfeats_cp_sk_nir, gfeats, qids_csn, gids, qimage_ids_csn, gimage_ids, mode,
            qfeats_cp_sk_nir_cp, qfeats_cp_sk_nir_sk, qfeats_cp_sk_nir_nir)
            # query_feats_1_tse=qfeats_cp_sk_nir_cp_tse, query_feats_2_tse=qfeats_cp_sk_nir_sk_tse, 
            # query_feats_3_tse=qfeats_cp_sk_nir_nir_tse, gallery_feats_tse=gfeats_tse,
            # query_feats_tse=qfeats_cp_sk_nir_tse)
        res14=self._compute_sim(qfeats_text_cp_sk, gfeats, qids_t_cs, gids, qimage_ids_tcs, gimage_ids, mode, 
            qfeats_text_cp_sk_cp, qfeats_text_cp_sk_sk, qfeats_text_cp_sk_text)
            # query_feats_1_tse=qfeats_text_cp_sk_cp_tse, query_feats_2_tse=qfeats_text_cp_sk_sk_tse, 
            # query_feats_3_tse=qfeats_text_cp_sk_text_tse, gallery_feats_tse=gfeats_tse,
            # query_feats_tse=qfeats_text_cp_sk_tse)


        if mode == 'train_val':
            threemodal_rank1 = (res11[0] + res12[0] + res13[0] + res14[0]) / 4
            threemodal_rank5 = (res11[1] + res12[1] + res13[1] + res14[1]) / 4
            threemodal_rank10 = (res11[2] + res12[2] + res13[2] + res14[2]) / 4
            threemodal_map = (res11[3] + res12[3] + res13[3] + res14[3]) / 4
            threemodal_minp = (res11[4] + res12[4] + res13[4] + res14[4]) / 4


        # ----------------四模态----------------
        qids_tcsn, qfeats_text_cp_sk_nir, qimage_ids_tcsn, qfeats_text_cp_sk_nir_cp, qfeats_text_cp_sk_nir_sk, \
            qfeats_text_cp_sk_nir_nir, qfeats_text_cp_sk_nir_text= self._fourmodal_iiit(
            self.text_cp_sk_nir_loader, device, model, InputImageType.sketch, InputImageType.color_pencil, InputImageType.nir)
        res15=self._compute_sim(qfeats_text_cp_sk_nir, gfeats, qids_tcsn, gids, qimage_ids_tcsn, gimage_ids, mode,
                qfeats_text_cp_sk_nir_cp, qfeats_text_cp_sk_nir_sk, qfeats_text_cp_sk_nir_nir, qfeats_text_cp_sk_nir_text)
                # query_feats_1_tse=qfeats_text_cp_sk_nir_cp_tse, query_feats_2_tse=qfeats_text_cp_sk_nir_sk_tse,
                # query_feats_3_tse=qfeats_text_cp_sk_nir_nir_tse, query_feats_4_tse=qfeats_text_cp_sk_nir_text_tse,
                # gallery_feats_tse=gfeats_tse,
                # query_feats_tse=qfeats_text_cp_sk_nir_tse)

        if mode == 'train_val':
            fourmodal_rank1 = res15[0]
            fourmodal_rank5 = res15[1]
            fourmodal_rank10 = res15[2]
            fourmodal_map = res15[3]
            fourmodal_minp = res15[4]

        if mode == 'test':
            return self.all_query_ids_list, self.all_gallery_ids_list

        return [onemodal_rank1, onemodal_rank5, onemodal_rank10, onemodal_map, onemodal_minp], \
                [twomodal_rank1, twomodal_rank5, twomodal_rank10, twomodal_map, twomodal_minp], \
                [threemodal_rank1, threemodal_rank5, threemodal_rank10, threemodal_map, threemodal_minp],\
                [fourmodal_rank1, fourmodal_rank5, fourmodal_rank10, fourmodal_map, fourmodal_minp],
        # return qfeats_text, qfeats_sketch, qfeats_text_sketch, gfeats, qids, qids_sketch, gids, \
        #     qimage_ids, qimage_ids_sketch, gimage_ids
    
    def _compute_sim(self, query_feats, gallery_feats, query_ids, gallery_ids, query_image_ids, gallery_image_ids, 
                     mode='train_val', query_feats_1=None, query_feats_2=None, query_feats_3=None, query_feats_4=None,
                    #  query_feats_tse=None, gallery_feats_tse = None, 
                    #  query_feats_1_tse=None, query_feats_2_tse=None, query_feats_3_tse=None, query_feats_4_tse=None
                     ):
        query_feats = F.normalize(query_feats, p=2, dim=1) # 
        gallery_feats = F.normalize(gallery_feats, p=2, dim=1)

        # if query_feats_tse is not None and gallery_feats_tse is not None:
        #     query_feats_tse = F.normalize(query_feats_tse, p=2, dim=1) # 
        #     gallery_feats_tse = F.normalize(gallery_feats_tse, p=2, dim=1)

        # print('111')
        
        if self.args.rerank:
            dist = re_ranking(query_feats, gallery_feats, k1=100, k2=30, lambda_value=0.3)
            print('finish one reranking !')
            sim = -dist
        else:
            ### tse 加入: 注意, 暂时没有加入fusion feature的tse feature
            # sim_tse = None
            # if query_feats_tse is not None and gallery_feats_tse is not None:
                # sim_tse = query_feats_tse @ gallery_feats_tse.t()
            sim = query_feats @ gallery_feats.t()
            # sim = (sim + sim_tse) / 2 if sim_tse is not None else sim
        q_num = 1

        # 最简单的思路：直接相加sim取平均; or 把所有query feats先取平均
        if query_feats_1 is not None:
            query_feats_1 = F.normalize(query_feats_1, p=2, dim=1)
            # query_feats_1_tse = F.normalize(query_feats_1_tse, p=2, dim=1)
            
            if self.args.rerank:
                sim_1 = -re_ranking(query_feats_1, gallery_feats, k1=100, k2=30, lambda_value=0.3)
            else:
                # sim_1_tse = query_feats_1_tse @ gallery_feats_tse.t()
                sim_1 = query_feats_1 @ gallery_feats.t()
                # sim_1 = (sim_1 + sim_1_tse) / 2
            sim += sim_1
            q_num += 1

        if query_feats_2 is not None:
            query_feats_2 = F.normalize(query_feats_2, p=2, dim=1)
            # query_feats_2_tse = F.normalize(query_feats_2_tse, p=2, dim=1)
            if self.args.rerank:
                sim_2 = -re_ranking(query_feats_2, gallery_feats, k1=100, k2=30, lambda_value=0.3)
            else:
                # sim_2_tse = query_feats_2_tse @ gallery_feats_tse.t()
                sim_2 = query_feats_2 @ gallery_feats.t()
                # sim_2 = (sim_2 + sim_2_tse) / 2
            sim += sim_2
            q_num += 1

        if query_feats_3 is not None:
            query_feats_3 = F.normalize(query_feats_3, p=2, dim=1)
            # query_feats_3_tse = F.normalize(query_feats_3_tse, p=2, dim=1)
            if self.args.rerank:
                sim_3 = -re_ranking(query_feats_3, gallery_feats, k1=100, k2=30, lambda_value=0.3)
            else:
                # sim_3_tse = query_feats_3_tse @ gallery_feats_tse.t()
                sim_3 = query_feats_3 @ gallery_feats.t()
                # sim_3 = (sim_3 + sim_3_tse) / 2
            sim += sim_3
            q_num += 1

        if query_feats_4 is not None:
            query_feats_4 = F.normalize(query_feats_4, p=2, dim=1)
            # query_feats_4_tse = F.normalize(query_feats_4_tse, p=2, dim=1)
            if self.args.rerank:
                sim_4 = -re_ranking(query_feats_4, gallery_feats, k1=100, k2=30, lambda_value=0.3)
            else:
                # sim_4_tse = query_feats_4_tse @ gallery_feats_tse.t()
                sim_4 = query_feats_4 @ gallery_feats.t()
                # sim_4 = (sim_4 + sim_4_tse) / 2
            sim += sim_4
            q_num += 1

        sim = sim / q_num

        # 测试
        if mode == 'test':
            sim = torch.tensor(sim).to(query_feats.device)
            _,indices = torch.topk(sim,k=100,dim=1,largest=True,sorted=True)
            for r in range(indices.shape[0]):
                self.all_query_ids_list.append(int(query_ids[r]))
                self.all_gallery_ids_list.append(list(map(int, indices[r].cpu() + 1)))  # indices是从0开始的,gallery最后是从1.jpg开始
            print('finish one query mode')
            return None
        
        if not self.args.rerank:
            sim = sim.detach().cpu().numpy()

        t2i_cmc, t2i_mAP, t2i_mINP = eval_func(
            -sim, query_ids.numpy(), gallery_ids.numpy(), # -sim.detach().cpu().numpy()
            query_image_ids.numpy(), gallery_image_ids.numpy(), set=0, max_rank=10)
        
        return t2i_cmc[0], t2i_cmc[4], t2i_cmc[9], t2i_mAP, t2i_mINP
    
    def _fourmodal_iiit(self, img_img_img_text_loader, device, model, input_img_type1=None, input_img_type2=None, input_img_type3=None):
        qids,qfeats_text,qfeats_text_img_img_img, qimage_ids, qimage_ids2=[],[],[],[],[]
        qfeats_img, qfeats_img2, qfeats_img3 = [], [], []

        # qfeats_text_tse, qfeats_img_tse, qfeats_img2_tse, qfeats_img3_tse = [], [], [], []
        # qfeats_text_img_img_img_tse = []
        for img, image_id, img2, image_id2,img3, image_id3,pid, caption in img_img_img_text_loader:
            caption = caption.to(device)
            img = img.to(device)
            img2 = img2.to(device)
            img3 = img3.to(device)
            with torch.no_grad():
                text_feat = model.encode_text(caption)
                img_feat = model.encode_image(img, input_img_type1)
                img2_feat = model.encode_image(img2, input_img_type2)
                img3_feat = model.encode_image(img3, input_img_type3)

                # text_feat_tse = model.encode_text_tse(caption)
                # img_feat_tse = model.encode_image_tse(img, input_img_type1)
                # img2_feat_tse = model.encode_image_tse(img2, input_img_type2)
                # img3_feat_tse = model.encode_image_tse(img3, input_img_type3)
                if self.args.fusion_way in ['add', 'weight add', 'cross attention', 'parameter add', 'concat', 'global concat', 'cross attention text', 'cross attention sketch', 'concat transformer']:
                    text_img_img_img_fu = model.fusion_layer(
                        text_feat, img_feat, img2_feat, img3_feat, caption, way=self.args.fusion_way
                        )
                
            qids.append(pid.view(-1)) # flatten 
            qfeats_text_img_img_img.append(text_img_img_img_fu)
            # qfeats_text_img_img_img_tse.append(text_img_img_img_fu_tse)
            qimage_ids.append(image_id)

            qfeats_text.append(text_feat[torch.arange(text_feat.shape[0]), caption.argmax(dim=-1)].float())
            qfeats_img.append(img_feat[:,0,:].float())
            qfeats_img2.append(img2_feat[:,0,:].float())
            qfeats_img3.append(img3_feat[:,0,:].float())
            # qfeats_text_tse.append(text_feat_tse)
            # qfeats_img_tse.append(img_feat_tse)
            # qfeats_img2_tse.append(img2_feat_tse)
            # qfeats_img3_tse.append(img3_feat_tse)

        qids = torch.cat(qids, 0)
        qfeats_text_img_img_img = torch.cat(qfeats_text_img_img_img, 0)
        # qfeats_text_img_img_img_tse = torch.cat(qfeats_text_img_img_img_tse, 0)
        qimage_ids = torch.cat(qimage_ids, 0)
        
        qfeats_text = torch.cat(qfeats_text, 0)
        qfeats_img = torch.cat(qfeats_img, 0)
        qfeats_img2 = torch.cat(qfeats_img2, 0)
        qfeats_img3 = torch.cat(qfeats_img3, 0)

        # qfeats_text_tse = torch.cat(qfeats_text_tse, 0)
        # qfeats_img_tse = torch.cat(qfeats_img_tse, 0)
        # qfeats_img2_tse = torch.cat(qfeats_img2_tse, 0)
        # qfeats_img3_tse = torch.cat(qfeats_img3_tse, 0)
        return qids, qfeats_text_img_img_img, qimage_ids, qfeats_img, qfeats_img2, qfeats_img3, qfeats_text#, \
                # qfeats_img_tse, qfeats_img2_tse, qfeats_img3_tse, qfeats_text_tse, qfeats_text_img_img_img_tse
    
    def _threemodal_iii(self, img_img_img_loader, device, model, input_img_type1=None, input_img_type2=None, input_img_type3=None):
        qids, qfeats_img_img_img, qimage_ids=[],[],[]
        qfeats_img, qfeats_img2, qfeats_img3 = [], [], []

        # qfeats_img_tse, qfeats_img2_tse, qfeats_img3_tse = [], [], []
        # qfeats_img_img_img_tse = []
        for pid, img, image_id, pid2, img2, image_id2, pid3, img3, image_id3 in img_img_img_loader:
            # assert pid == pid2
            # 按理说 img id 和 pid 应该相同
            img3 = img3.to(device)
            img2 = img2.to(device)
            img = img.to(device)
            with torch.no_grad():
                img3_feat = model.encode_image(img3, input_img_type3)
                img2_feat = model.encode_image(img2, input_img_type2)
                img_feat = model.encode_image(img, input_img_type1)

                # img3_feat_tse = model.encode_image_tse(img3, input_img_type3)
                # img2_feat_tse = model.encode_image_tse(img2, input_img_type2)
                # img_feat_tse = model.encode_image_tse(img, input_img_type1)
                if self.args.fusion_way in ['add', 'weight add', 'cross attention', 'parameter add', 'concat', 
                                            'global concat', 'cross attention text', 'cross attention sketch', 'concat transformer']:
                    img_img_img_fu = model.fusion_layer(
                        None, img_feat, img2_feat, img3_feat, None, way=self.args.fusion_way
                        )

                
            qids.append(pid.view(-1)) # flatten 
            # qfeats_img.append(img_feat)
            qfeats_img_img_img.append(img_img_img_fu)
            # qfeats_img_img_img_tse.append(img_img_img_fu_tse)

            qimage_ids.append(image_id)

            qfeats_img.append(img_feat[:,0,:].float())
            qfeats_img2.append(img2_feat[:,0,:].float())
            qfeats_img3.append(img3_feat[:,0,:].float())

            # qfeats_img_tse.append(img_feat_tse)
            # qfeats_img2_tse.append(img2_feat_tse)
            # qfeats_img3_tse.append(img3_feat_tse)
            

        qids = torch.cat(qids, 0)
        # qfeats_text = torch.cat(qfeats_text, 0)
        qfeats_img_img_img = torch.cat(qfeats_img_img_img, 0)
        # qfeats_img_img_img_tse = torch.cat(qfeats_img_img_img_tse, 0)

        qimage_ids = torch.cat(qimage_ids, 0)
        
        qfeats_img = torch.cat(qfeats_img, 0)
        qfeats_img2 = torch.cat(qfeats_img2, 0)
        qfeats_img3 = torch.cat(qfeats_img3, 0)

        # qfeats_img_tse = torch.cat(qfeats_img_tse, 0)
        # qfeats_img2_tse = torch.cat(qfeats_img2_tse, 0)
        # qfeats_img3_tse = torch.cat(qfeats_img3_tse, 0)

        return qids, qfeats_img_img_img, qimage_ids, qfeats_img, qfeats_img2, qfeats_img3#, \
                # qfeats_img_tse, qfeats_img2_tse, qfeats_img3_tse, qfeats_img_img_img_tse3
    
    def _threemodal_iit(self, img_img_text_loader, device, model, input_img_type1=None, input_img_type2=None):
        qids,qfeats_text,qfeats_text_img_img, qimage_ids, qimage_ids2=[],[],[],[],[]
        qfeats_img, qfeats_img2 = [], []

        # qfeats_text_tse, qfeats_img_tse, qfeats_img2_tse = [], [], []
        # qfeats_text_img_img_tse = []
        for img, image_id, img2, image_id2,pid, caption in img_img_text_loader:
            caption = caption.to(device)
            img = img.to(device)
            img2 = img2.to(device)
            with torch.no_grad():
                text_feat = model.encode_text(caption)
                img_feat = model.encode_image(img, input_img_type1)
                img2_feat = model.encode_image(img2, input_img_type2)

                # text_feat_tse = model.encode_text_tse(caption)
                # img_feat_tse = model.encode_image_tse(img, input_img_type1)
                # img2_feat_tse = model.encode_image_tse(img2, input_img_type2)

                if self.args.fusion_way in ['add', 'weight add', 'cross attention', 
                                            'parameter add', 'concat', 'global concat', 'cross attention text', 'cross attention sketch', 'concat transformer']:
                    
                    ########### 发现大bug: 这个应该按照input image type选择位置啊 !!!!!! 0806#
                    # 修复
                    if input_img_type1 == InputImageType.sketch and input_img_type2 == InputImageType.color_pencil:
                        text_img_img_fu = model.fusion_layer(
                            text_feat, img_feat, img2_feat, None, caption, way=self.args.fusion_way
                            )
                    elif input_img_type1 == InputImageType.sketch and input_img_type2 == InputImageType.nir:
                        text_img_img_fu = model.fusion_layer(
                            text_feat, img_feat, None, img2_feat, caption, way=self.args.fusion_way
                            )
                    elif input_img_type1 == InputImageType.color_pencil and input_img_type2 == InputImageType.nir:
                        text_img_img_fu = model.fusion_layer(
                            text_feat, None, img_feat, img2_feat, caption, way=self.args.fusion_way
                            )
                
            qids.append(pid.view(-1)) # flatten 
            # qfeats_text.append(text_feat)
            qfeats_text_img_img.append(text_img_img_fu)
            # qfeats_text_img_img_tse.append(text_img_img_fu_tse)

            qimage_ids.append(image_id)
            # qimage_ids2.append(image_id2)
            qfeats_img.append(img_feat[:,0,:].float())
            qfeats_img2.append(img2_feat[:,0,:].float())
            qfeats_text.append(text_feat[torch.arange(text_feat.shape[0]), caption.argmax(dim=-1)].float())
            # qfeats_text_tse.append(text_feat_tse)
            # qfeats_img_tse.append(img_feat_tse)
            # qfeats_img2_tse.append(img2_feat_tse)

        qids = torch.cat(qids, 0)
        qfeats_text_img_img = torch.cat(qfeats_text_img_img, 0)
        # qfeats_text_img_img_tse = torch.cat(qfeats_text_img_img_tse, 0)
        qimage_ids = torch.cat(qimage_ids, 0)

        qfeats_text = torch.cat(qfeats_text, 0)
        qfeats_img = torch.cat(qfeats_img, 0)
        qfeats_img2 = torch.cat(qfeats_img2, 0)
        # qfeats_text_tse = torch.cat(qfeats_text_tse, 0)
        # qfeats_img_tse = torch.cat(qfeats_img_tse, 0)
        # qfeats_img2_tse = torch.cat(qfeats_img2_tse, 0)
        
        return qids, qfeats_text_img_img, qimage_ids, qfeats_img, qfeats_img2, qfeats_text#, \
                # qfeats_img_tse,qfeats_img2_tse, qfeats_text_tse, qfeats_text_img_img_tse
    
    def _twomodal_it(self, img_text_loader, device, model, input_img_type=None):
        qids,qfeats_text_img, qimage_ids=[],[],[]
        qfeats_img, qfeats_text = [], []

        # qfeats_img_tse, qfeats_text_tse = [], []
        # qfeats_text_img_tse = []
        for img, image_id, pid, caption in img_text_loader:
            caption = caption.to(device)
            img = img.to(device)
            with torch.no_grad():
                text_feat = model.encode_text(caption)
                img_feat = model.encode_image(img, input_img_type)

                if self.args.fusion_way in ['add', 'weight add', 'cross attention', 
                                            'parameter add', 'concat', 'global concat', 'cross attention text', 'cross attention sketch', 'concat transformer']:
                    if input_img_type == InputImageType.sketch:
                        text_img_text_fu = model.fusion_layer(
                            text_feat, img_feat, None, None, caption, way=self.args.fusion_way)
                    elif input_img_type == InputImageType.color_pencil:
                        text_img_text_fu = model.fusion_layer(
                            text_feat, None, img_feat, None, caption, way=self.args.fusion_way)
                    elif input_img_type == InputImageType.nir:
                        text_img_text_fu = model.fusion_layer(
                            text_feat, None, None, img_feat, caption, way=self.args.fusion_way)
                
            qids.append(pid.view(-1)) # flatten 
            # qfeats_text.append(text_feat)
            qfeats_text_img.append(text_img_text_fu)
            # qfeats_text_img_tse.append(text_img_text_fu_tse)
            qimage_ids.append(image_id)

            qfeats_img.append(img_feat[:,0,:].float())
            qfeats_text.append(text_feat[torch.arange(text_feat.shape[0]), caption.argmax(dim=-1)].float())
            # qfeats_img_tse.append(img_feat_tse)
            # qfeats_text_tse.append(text_feat_tse)
            

        qids = torch.cat(qids, 0)
        # qfeats_text = torch.cat(qfeats_text, 0)
        qfeats_text_img = torch.cat(qfeats_text_img, 0)
        # qfeats_text_img_tse = torch.cat(qfeats_text_img_tse, 0)
        qimage_ids = torch.cat(qimage_ids, 0)

        qfeats_img = torch.cat(qfeats_img, 0)
        qfeats_text = torch.cat(qfeats_text, 0)
        # qfeats_img_tse = torch.cat(qfeats_img_tse, 0)
        # qfeats_text_tse = torch.cat(qfeats_text_tse, 0)

        return qids, qfeats_text_img, qimage_ids, qfeats_img, qfeats_text#, qfeats_img_tse, qfeats_text_tse, qfeats_text_img_tse
    

    def _twomodal_ii(self, img_img_loader, device, model, input_img_type1=None, input_img_type2=None):
        qids, qfeats_img_img, qimage_ids=[],[],[]
        qfeats_img, qfeats_img2 = [], []

        # qfeats_img_tse, qfeats_img2_tse = [], []
        # qfeats_img_img_tse = []
        for pid, img, image_id, pid2, img2, image_id2 in img_img_loader:
            # assert pid == pid2
            img2 = img2.to(device)
            img = img.to(device)
            with torch.no_grad():
                img2_feat = model.encode_image(img2, input_img_type2)
                img_feat = model.encode_image(img, input_img_type1)

                # img2_feat_tse = model.encode_image_tse(img2, input_img_type2)
                # img_feat_tse = model.encode_image_tse(img, input_img_type1)

                if self.args.fusion_way in ['add', 'weight add', 'cross attention', 
                                            'parameter add', 'concat', 'global concat', 'cross attention text', 'cross attention sketch', 'concat transformer']:
                    if input_img_type1 == InputImageType.sketch and input_img_type2 == InputImageType.color_pencil:
                        img_img_fu = model.fusion_layer(
                            None, img_feat, img2_feat, None, None, way=self.args.fusion_way)
                    elif input_img_type1 == InputImageType.sketch and input_img_type2 == InputImageType.nir:
                        img_img_fu = model.fusion_layer(
                            None, img_feat, None, img2_feat, None, way=self.args.fusion_way)
                    elif input_img_type1 == InputImageType.color_pencil and input_img_type2 == InputImageType.nir:
                        img_img_fu = model.fusion_layer(
                            None, None, img_feat, img2_feat, None, way=self.args.fusion_way)
                
            qids.append(pid.view(-1)) # flatten 
            # qfeats_img.append(img_feat)
            qfeats_img_img.append(img_img_fu)
            # qfeats_img_img_tse.append(img_img_fu_tse)
            qimage_ids.append(image_id)

            qfeats_img.append(img_feat[:,0,:].float())
            qfeats_img2.append(img2_feat[:,0,:].float())
            # qfeats_img_tse.append(img_feat_tse)
            # qfeats_img2_tse.append(img2_feat_tse)
            

        qids = torch.cat(qids, 0)
        # qfeats_text = torch.cat(qfeats_text, 0)
        qfeats_img_img = torch.cat(qfeats_img_img, 0)
        # qfeats_img_img_tse = torch.cat(qfeats_img_img_tse, 0)
        qimage_ids = torch.cat(qimage_ids, 0)

        qfeats_img = torch.cat(qfeats_img, 0)
        qfeats_img2 = torch.cat(qfeats_img2, 0)
        # qfeats_img_tse = torch.cat(qfeats_img_tse, 0)
        # qfeats_img2_tse = torch.cat(qfeats_img2_tse, 0)
        return qids, qfeats_img_img, qimage_ids, qfeats_img, qfeats_img2#, qfeats_img_tse, qfeats_img2_tse, qfeats_img_img_tse


    def _onemodal(self, data_loader, device, model, input_img_type=None):
        qids, qfeats_img, qimage_ids=[],[],[]
        # qfeats_img_tse = []

        for pid, img, image_id in data_loader:
            img = img.to(device)
            with torch.no_grad():
                img_feat = model.encode_image(img, input_img_type)[:, 0, :].float()
                # img_feat_tse = model.encode_image_tse(img, input_img_type)
            qids.append(pid.view(-1)) # flatten
            qfeats_img.append(img_feat)

            # qfeats_img_tse.append(img_feat_tse)
            qimage_ids.append(image_id)

        qids = torch.cat(qids, 0)
        qfeats_img = torch.cat(qfeats_img, 0)
        # qfeats_img_tse = torch.cat(qfeats_img_tse, 0)
        qimage_ids = torch.cat(qimage_ids, 0)
        return qids, qfeats_img, qimage_ids#, qfeats_img_tse


    def _onemodal_t(self, data_loader, device, model):
        qids, qfeats_text, qtext_ids=[],[],[]
        # qfeats_text_tse = []
        for pid, caption, caption_id in data_loader:
            caption = caption.to(device)
            with torch.no_grad():
                text_feat = model.encode_text(caption)
                # text_feat_tse = model.encode_text_tse(caption)

                text_feat = text_feat[torch.arange(text_feat.shape[0]), caption.argmax(dim=-1)].float()
            qids.append(pid.view(-1)) # flatten
            qfeats_text.append(text_feat) 
            # tse embedding
            # qfeats_text_tse.append(text_feat_tse)

            qtext_ids.append(caption_id) 

        qids = torch.cat(qids, 0)
        qfeats_text = torch.cat(qfeats_text, 0)
        # qfeats_text_tse = torch.cat(qfeats_text_tse, 0)
        qtext_ids = torch.cat(qtext_ids, 0)

        return qids, qfeats_text, qtext_ids#, qfeats_text_tse

    def _show_table(self, onemodal_list, twomodal_list, threemodal_list, fourmodal_list):
        table = PrettyTable(["task", "R1", "R5", "R10", "mAP", "mINP"])
        table.add_row(['t2i-one_modal_RGB', onemodal_list[0], onemodal_list[1], onemodal_list[2], onemodal_list[3], onemodal_list[4]])
        table.add_row(['t2i-two_modal_RGB', twomodal_list[0], twomodal_list[1], twomodal_list[2], twomodal_list[3], twomodal_list[4]])
        table.add_row(['t2i-three_modal_RGB', threemodal_list[0], threemodal_list[1], threemodal_list[2], threemodal_list[3], threemodal_list[4]])
        table.add_row(['t2i-four_modal_RGB', fourmodal_list[0], fourmodal_list[1], fourmodal_list[2], fourmodal_list[3], fourmodal_list[4]])
        # table.add_row(['t2i-text_RGB', t2i_cmc[0], t2i_cmc[4], t2i_cmc[9], t2i_mAP, t2i_mAP])

        table.float_format = '.4'
        self.logger.info('\n' + str(table))

    
    def eval(self, model, i2t_metric=False, mode='train_val'):

        if mode == 'test':
            return self._compute_embedding(model, mode)

        onemodal_list, twomodal_list, threemodal_list, fourmodal_list = self._compute_embedding(model, mode)
        self._show_table(onemodal_list, twomodal_list, threemodal_list, fourmodal_list)

        return onemodal_list[3], twomodal_list[3], threemodal_list[3], fourmodal_list[3]  ### 以rank-1作为标准(还是改为map, rank-1都是100)
    
        # qfeats_text, qfeats_sketch, qfeats_text_sketch, gfeats, qids, qids_sketch, gids, qimage_ids, qimage_ids_sketch, gimage_ids = self._compute_embedding(model)

        # qfeats_text = F.normalize(qfeats_text, p=2, dim=1) # text features
        # qfeats_sketch = F.normalize(qfeats_sketch, p=2, dim=1) # sketch features
        # qfeats_text_sketch = F.normalize(qfeats_text_sketch, p=2, dim=1) # sketch+text features

        # gfeats = F.normalize(gfeats, p=2, dim=1) # image features

        # similarity_text_rgb = qfeats_text @ gfeats.t()
        # similarity_sketch_rgb = qfeats_sketch @ gfeats.t()
        # similarity_textsketch_rgb = qfeats_text_sketch @ gfeats.t()
        
        # #original gallery set for text-to-rgb retrieval
        # t2i_cmc, t2i_mAP, t2i_mINP = eval_func(-similarity_text_rgb.detach().cpu().numpy() , qids.numpy(), gids.numpy(), qimage_ids.numpy(), gimage_ids.numpy(), set=0, max_rank=10)
        
        # # remove the rgb images that used for generated sketches from gallery set
        # t2i_cmc0, t2i_mAP0, t2i_mINP0 = eval_func(-similarity_text_rgb.detach().cpu().numpy() , qids.numpy(), gids.numpy(), qimage_ids.numpy(), gimage_ids.numpy(), set=2, max_rank=10)
        
        # t2i_cmc1, t2i_mAP1, t2i_mINP1 = eval_func(-similarity_sketch_rgb.detach().cpu().numpy() , qids_sketch.numpy(), gids.numpy(), qimage_ids_sketch.numpy(), gimage_ids.numpy(), set=2, max_rank=10)
        # t2i_cmc2, t2i_mAP2, t2i_mINP2 = eval_func(-similarity_textsketch_rgb.detach().cpu().numpy() , qids.numpy(), gids.numpy(), qimage_ids.numpy(), gimage_ids.numpy(), set=2, max_rank=10)

        # table = PrettyTable(["task", "R1", "R5", "R10", "mAP", "mINP"])
        # table.add_row(['t2i-text_RGB_original', t2i_cmc[0], t2i_cmc[4], t2i_cmc[9], t2i_mAP, t2i_mINP])
        # table.add_row(['t2i-text_RGB', t2i_cmc0[0], t2i_cmc0[4], t2i_cmc0[9], t2i_mAP0, t2i_mINP0])
        # table.add_row(['t2i-sketch_RGB', t2i_cmc1[0], t2i_cmc1[4], t2i_cmc1[9], t2i_mAP1, t2i_mINP1])
        # table.add_row(['t2i-textsketch_RGB', t2i_cmc2[0], t2i_cmc2[4], t2i_cmc2[9], t2i_mAP2, t2i_mINP2])
        # # table.add_row(['t2i-text_RGB', t2i_cmc[0], t2i_cmc[4], t2i_cmc[9], t2i_mAP, t2i_mAP])

        # if i2t_metric:
        #     i2t_cmc, i2t_mAP, _ = rank(similarity=similarity_text_rgb.t(), q_pids=gids, g_pids=qids, max_rank=10, get_mAP=True)
        #     i2t_cmc, i2t_mAP = i2t_cmc.cpu().numpy(), i2t_mAP.cpu().numpy()
        #     table.add_row(['i2t', i2t_cmc[0], i2t_cmc[4], i2t_cmc[9], i2t_mAP])

        # table.float_format = '.4'
        # self.logger.info('\n' + str(table))
        
        # return t2i_cmc[0], t2i_cmc1[0], t2i_cmc2[0]
    

    # def eval_by_proj(self, model, i2t_metric=False):

    #     qfeats, gfeats, qids, gids = self._compute_embedding(model)

    #     # qfeats_norm = F.normalize(qfeats, p=2, dim=1) # text features
    #     gfeats_norm = F.normalize(gfeats, p=2, dim=1) # image features

    #     similarity = qfeats @ gfeats_norm.t()

    #     t2i_cmc, t2i_mAP, _ = rank(similarity=similarity, q_pids=qids, g_pids=gids, max_rank=10, get_mAP=True)
    #     t2i_cmc, t2i_mAP = t2i_cmc.cpu().numpy(), t2i_mAP.cpu().numpy()
    #     table = PrettyTable(["task", "R1", "R5", "R10", "mAP"])
    #     table.add_row(['t2i', t2i_cmc[0], t2i_cmc[4], t2i_cmc[9], t2i_mAP])

    #     if i2t_metric:
    #         i2t_cmc, i2t_mAP, _ = rank(similarity=similarity.t(), q_pids=gids, g_pids=qids, max_rank=10, get_mAP=True)
    #         i2t_cmc, i2t_mAP = i2t_cmc.cpu().numpy(), i2t_mAP.cpu().numpy()
    #         table.add_row(['i2t', i2t_cmc[0], i2t_cmc[4], i2t_cmc[9], i2t_mAP])
    #     table.float_format = '.4'
    #     self.logger.info('\n' + str(table))
        
    #     return t2i_cmc[0]

