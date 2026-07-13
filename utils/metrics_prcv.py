from prettytable import PrettyTable
import torch
import numpy as np
import os
import torch.nn.functional as F
import logging
from model.utils import InputImageType
from utils.compute_dist import build_dist
from tqdm import tqdm

def rank(similarity, q_pids, g_pids, max_rank=10, get_mAP=True):
    if get_mAP:
        indices = torch.argsort(similarity, dim=1, descending=True)
    else:
        # acclerate sort with topk
        _, indices = torch.topk(
            similarity, k=max_rank, dim=1, largest=True, sorted=True
        )  # q * topk
    indices = indices.to(g_pids.device)
    pred_labels = g_pids[indices]  # q * k
    matches = pred_labels.eq(q_pids.view(-1, 1))  # q * k

    all_cmc = matches[:, :max_rank].cumsum(1)  # cumulative sum
    all_cmc[all_cmc > 1] = 1
    all_cmc = all_cmc.float().mean(0) * 100
    # all_cmc = all_cmc[topk - 1]

    if not get_mAP:
        return all_cmc, torch.tensor(0), torch.tensor(0), indices

    num_rel = matches.sum(1)  # q
    tmp_cmc = matches.cumsum(1)  # q * k

    inp = [tmp_cmc[i][match_row.nonzero()[-1]] / (match_row.nonzero()[-1] + 1.) for i, match_row in enumerate(matches)]
    mINP = torch.cat(inp).mean() * 100

    tmp_cmc = [tmp_cmc[:, i] / (i + 1.0) for i in range(tmp_cmc.shape[1])]
    tmp_cmc = torch.stack(tmp_cmc, 1) * matches
    AP = tmp_cmc.sum(1) / num_rel  # q
    mAP = AP.mean() * 100

    return all_cmc.numpy(), mAP.numpy(), mINP.numpy(), indices

# def rank(similarity, q_pids, g_pids, max_rank=10, get_mAP=True):
#     if get_mAP:
#         indices = torch.argsort(similarity, dim=1, descending=True)
#     else:
#         # acclerate sort with topk
#         _, indices = torch.topk(
#             similarity, k=max_rank, dim=1, largest=True, sorted=True
#         )  # q * topk
#     pred_labels = g_pids[indices]  # q * k
#     matches = pred_labels.eq(q_pids.view(-1, 1))  # q * k

#     all_cmc = matches[:, :max_rank].cumsum(1) # cumulative sum
#     all_cmc[all_cmc > 1] = 1
#     all_cmc = all_cmc.float().mean(0) * 100
#     # all_cmc = all_cmc[topk - 1]

#     if not get_mAP:
#         return all_cmc, indices

#     num_rel = matches.sum(1)  # q
#     tmp_cmc = matches.cumsum(1)  # q * k
#     tmp_cmc = [tmp_cmc[:, i] / (i + 1.0) for i in range(tmp_cmc.shape[1])]
#     tmp_cmc = torch.stack(tmp_cmc, 1) * matches
#     AP = tmp_cmc.sum(1) / num_rel  # q
#     mAP = AP.mean() * 100
#     return all_cmc, mAP, indices


# def eval_func(distmat, q_pids, g_pids, q_camids, g_camids, set=0, max_rank=50):
#     """Evaluation with market1501 metric
#         Key: for each query identity, its gallery images from the same camera view are discarded.
#         """
#     num_q, num_g = distmat.shape
#     if num_g < max_rank:
#         max_rank = num_g
#         print("Note: number of gallery samples is quite small, got {}".format(num_g))
#     indices = np.argsort(distmat, axis=1)
#     matches = (g_pids[indices] == q_pids[:, np.newaxis]).astype(np.int32)

#     # compute cmc curve for each query
#     all_cmc = []
#     all_AP = []
#     all_INP = []
#     num_valid_q = 0.  # number of valid query
#     for q_idx in range(num_q):
#         # get query pid and camid
#         q_pid = q_pids[q_idx]
#         q_camid = q_camids[q_idx]

#         # remove gallery samples that have the same pid and camid with query
#         if set == 2:
#             order = indices[q_idx]
#             remove = (g_pids[order] == q_pid) & (g_camids[order] == q_camid)
#             keep = np.invert(remove)

#             # compute cmc curve
#             # binary vector, positions with value 1 are correct matches
#             orig_cmc = matches[q_idx][keep]
#         else:
#             orig_cmc = matches[q_idx]

#         if not np.any(orig_cmc):
#             # this condition is true when query identity does not appear in gallery
#             continue


#         cmc = orig_cmc.cumsum()

#         pos_idx = np.where(orig_cmc == 1)
#         max_pos_idx = np.max(pos_idx)
#         inp = cmc[max_pos_idx]/ (max_pos_idx + 1.0)
#         all_INP.append(inp)

#         cmc[cmc > 1] = 1

#         all_cmc.append(cmc[:max_rank])
#         num_valid_q += 1.

#         # compute average precision
#         # reference: https://en.wikipedia.org/wiki/Evaluation_measures_(information_retrieval)#Average_precision
#         num_rel = orig_cmc.sum()
#         tmp_cmc = orig_cmc.cumsum()
#         tmp_cmc = [x / (i + 1.) for i, x in enumerate(tmp_cmc)]
#         tmp_cmc = np.asarray(tmp_cmc) * orig_cmc
#         AP = tmp_cmc.sum() / num_rel
#         all_AP.append(AP)

#     assert num_valid_q > 0, "Error: all query identities do not appear in gallery"

#     all_cmc = np.asarray(all_cmc).astype(np.float32)
#     all_cmc = all_cmc.sum(0) / num_valid_q
#     mAP = np.mean(all_AP)
#     mINP = np.mean(all_INP)

#     return all_cmc* 100, mAP* 100, mINP* 100

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
        self.logger = logging.getLogger("ORBench.eval")

        self.all_query_ids_list = []
        self.all_gallery_ids_list = []

    def _compute_embedding(self, model, mode='train_val'):
        """测试过程: 可以尝试返回单模态特征, 最后检索用单模态 + fusion 模态,  共同计算检索结果, 再投票一下; 
        只用到了 cls , eos  token, 没用融合特征啊 !!!! ---> 效果似乎并不好
        """
        
        model = model.eval()
        device = next(model.parameters()).device
        #gallery
        gids, gfeats = self._onemodal(self.img_loader, device, model, InputImageType.rgb)
        # ----------------单模态----------------
        #sk,cp,nir
        qids_sketch, qfeats_sketch = self._onemodal(self.sketch_loader, device, model, InputImageType.sketch)
        qids_cp, qfeats_cp = self._onemodal(self.cp_loader, device, model, InputImageType.color_pencil)
        qids_nir, qfeats_nir = self._onemodal(self.nir_loader, device, model, InputImageType.nir)
        qids_text, qfeats_text = self._onemodal_t(self.text_loader, device, model)

        # onemodal_rank1, onekodal_rank5, onemodal_rank10, onemodal_map,onemodal_minp = 0,0,0,0,0
        res1=self._compute_sim(qfeats_text, gfeats, qids_text, gids, None, None, mode)
        res2=self._compute_sim(qfeats_sketch, gfeats, qids_sketch, gids, None, None, mode)
        res3=self._compute_sim(qfeats_cp, gfeats, qids_cp, gids, None, None, mode)
        res4=self._compute_sim(qfeats_nir, gfeats, qids_nir, gids, None, None, mode)

        if mode == 'train_val':
            onemodal_rank1 = (res1[0] + res2[0] + res3[0] + res4[0]) / 4
            onemodal_rank5 = (res1[1] + res2[1] + res3[1] + res4[1]) / 4
            onemodal_rank10 = (res1[2] + res2[2] + res3[2] + res4[2]) / 4
            onemodal_map = (res1[3] + res2[3] + res3[3] + res4[3]) / 4
            onemodal_minp = (res1[4] + res2[4] + res3[4] + res4[4]) / 4

        # ----------------双模态----------------

        # text+sketch, tc, tn, sc, sn, cn
        qids_t_sk, qfeats_text_sk, qfeats_text_sk_sk, qfeats_text_sk_text = self._twomodal_it(
            self.text_sk_loader, device, model, InputImageType.sketch)
        qids_t_cp, qfeats_text_cp, qfeats_text_cp_cp, qfeats_text_cp_text = self._twomodal_it(
            self.text_cp_loader, device, model, InputImageType.color_pencil)
        qids_t_nir, qfeats_text_nir, qfeats_text_nir_nir, qfeats_text_nir_text = self._twomodal_it(
            self.text_nir_loader, device, model, InputImageType.nir)
        qids_sk_cp, qfeats_sk_cp, qfeats_sk_cp_sk, qfeats_sk_cp_cp = self._twomodal_ii(
            self.sk_cp_loader, device, model, InputImageType.sketch, InputImageType.color_pencil)
        qids_cp_nir, qfeats_cp_nir, qfeats_cp_nir_cp, qfeats_cp_nir_nir= self._twomodal_ii(
            self.cp_nir_loader, device, model, InputImageType.color_pencil, InputImageType.nir)
        qids_sk_nir, qfeats_sk_nir, qfeats_sk_nir_sk, qfeats_sk_nir_nir= self._twomodal_ii(
            self.sk_nir_loader, device, model, InputImageType.sketch, InputImageType.nir)
        res5=self._compute_sim(
            qfeats_text_sk, gfeats, qids_t_sk, gids, None, None, mode, qfeats_text_sk_sk, qfeats_text_sk_text)
        res6=self._compute_sim(
            qfeats_text_cp, gfeats, qids_t_cp, gids, None, None, mode, qfeats_text_cp_cp, qfeats_text_cp_text)
        res7=self._compute_sim(
            qfeats_text_nir, gfeats, qids_t_nir, gids, None, None, mode, qfeats_text_nir_nir, qfeats_text_nir_text)
        res8=self._compute_sim(
            qfeats_sk_cp, gfeats, qids_sk_cp, gids, None, None, mode, qfeats_sk_cp_sk, qfeats_sk_cp_cp)
        res9=self._compute_sim(
            qfeats_cp_nir, gfeats, qids_cp_nir, gids, None, None, mode, qfeats_cp_nir_cp, qfeats_cp_nir_nir)
        res10=self._compute_sim(
            qfeats_sk_nir, gfeats, qids_sk_nir, gids, None, None, mode, qfeats_sk_nir_sk, qfeats_sk_nir_nir)
        if mode == 'train_val':
            twomodal_rank1 = (res5[0] + res6[0] + res7[0] + res8[0] + res9[0] + res10[0]) / 6
            twomodal_rank5 = (res5[1] + res6[1] + res7[1] + res8[1] + res9[1] + res10[1]) / 6
            twomodal_rank10 = (res5[2] + res6[2] + res7[2] + res8[2] + res9[2] + res10[2]) / 6
            twomodal_map = (res5[3] + res6[3] + res7[3] + res8[3] + res9[3] + res10[3]) / 6
            twomodal_minp = (res5[4] + res6[4] + res7[4] + res8[4] + res9[4] + res10[4]) / 6

        # ----------------三模态----------------
        qids_t_cn, qfeats_text_cp_nir, qfeats_text_cp_nir_cp, qfeats_text_cp_nir_nir, qfeats_text_cp_nir_text= self._threemodal_iit(
            self.text_cp_nir_loader, device, model, InputImageType.color_pencil, InputImageType.nir)
        qids_t_sn, qfeats_text_sk_nir, qfeats_text_sk_nir_sk, qfeats_text_sk_nir_nir, qfeats_text_sk_nir_text= self._threemodal_iit(
            self.text_sk_nir_loader, device, model, InputImageType.sketch, InputImageType.nir)
        qids_t_cs, qfeats_text_cp_sk, qfeats_text_cp_sk_cp, qfeats_text_cp_sk_sk, qfeats_text_cp_sk_text= self._threemodal_iit(
            self.text_cp_sk_loader, device, model, InputImageType.color_pencil, InputImageType.sketch)  ### 注意顺序 !!!!!
        qids_csn, qfeats_cp_sk_nir, qfeats_cp_sk_nir_cp, qfeats_cp_sk_nir_sk, qfeats_cp_sk_nir_nir= self._threemodal_iii(
            self.cp_sk_nir_loader, device, model, InputImageType.color_pencil, InputImageType.sketch, InputImageType.nir)  ### 注意顺序 !!!!!
        res11=self._compute_sim(
            qfeats_text_cp_nir, gfeats, qids_t_cn, gids, None, None, mode, 
            qfeats_text_cp_nir_cp, qfeats_text_cp_nir_nir, qfeats_text_cp_nir_text)
        res12=self._compute_sim(qfeats_text_sk_nir, gfeats, qids_t_sn, gids, None, None, mode,
            qfeats_text_sk_nir_sk, qfeats_text_sk_nir_nir, qfeats_text_sk_nir_text)
        res13=self._compute_sim(qfeats_cp_sk_nir, gfeats, qids_csn, gids, None, None, mode,
            qfeats_cp_sk_nir_cp, qfeats_cp_sk_nir_sk, qfeats_cp_sk_nir_nir)
        res14=self._compute_sim(qfeats_text_cp_sk, gfeats, qids_t_cs, gids, None, None, mode, 
            qfeats_text_cp_sk_cp, qfeats_text_cp_sk_sk, qfeats_text_cp_sk_text)

        if mode == 'train_val':
            threemodal_rank1 = (res11[0] + res12[0] + res13[0] + res14[0]) / 4
            threemodal_rank5 = (res11[1] + res12[1] + res13[1] + res14[1]) / 4
            threemodal_rank10 = (res11[2] + res12[2] + res13[2] + res14[2]) / 4
            threemodal_map = (res11[3] + res12[3] + res13[3] + res14[3]) / 4
            threemodal_minp = (res11[4] + res12[4] + res13[4] + res14[4]) / 4


        # ----------------四模态----------------
        qids_tcsn, qfeats_text_cp_sk_nir, qfeats_text_cp_sk_nir_cp, qfeats_text_cp_sk_nir_sk, \
            qfeats_text_cp_sk_nir_nir, qfeats_text_cp_sk_nir_text= self._fourmodal_iiit(
            self.text_cp_sk_nir_loader, device, model, InputImageType.color_pencil, InputImageType.sketch, InputImageType.nir)
        res15=self._compute_sim(qfeats_text_cp_sk_nir, gfeats, qids_tcsn, gids, None, None, mode,
                qfeats_text_cp_sk_nir_cp, qfeats_text_cp_sk_nir_sk, qfeats_text_cp_sk_nir_nir, qfeats_text_cp_sk_nir_text)

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
    
    def _compute_sim(self, query_feats, gallery_feats, query_ids, gallery_ids, query_image_ids, gallery_image_ids, 
                     mode='train_val', query_feats_1=None, query_feats_2=None, query_feats_3=None, query_feats_4=None,
                     ):
        query_feats = F.normalize(query_feats.float(), p=2, dim=1)
        gallery_feats = F.normalize(gallery_feats.float(), p=2, dim=1)
        
        if self.args.rerank:
            # dist = re_ranking(query_feats, gallery_feats, k1=100, k2=30, lambda_value=0.3)
            dist_1 = build_dist(query_feats, gallery_feats, "euclidean")
            rerank_dist = build_dist(query_feats, gallery_feats, metric="jaccard", k1=100, k2=30)
            dist = rerank_dist * (1 - 0.3) + dist_1 * 0.3
            self.logger.info('finish one reranking !')
            sim = -dist
        else:
            sim = query_feats @ gallery_feats.t()
        q_num = 1

        if query_feats_1 is not None:
            query_feats_1 = F.normalize(query_feats_1.float(), p=2, dim=1)
            if self.args.rerank:
                # sim_1 = -re_ranking(query_feats_1, gallery_feats, k1=100, k2=30, lambda_value=0.3)
                dist_1 = build_dist(query_feats_1, gallery_feats, "euclidean")
                rerank_dist = build_dist(query_feats_1, gallery_feats, metric="jaccard", k1=100, k2=30)
                dist = rerank_dist * (1 - 0.3) + dist_1 * 0.3
                sim_1 = -dist
            else:
                sim_1 = query_feats_1 @ gallery_feats.t()
            sim += sim_1
            q_num += 1

        if query_feats_2 is not None:
            query_feats_2 = F.normalize(query_feats_2.float(), p=2, dim=1)
            if self.args.rerank:
                # sim_2 = -re_ranking(query_feats_2, gallery_feats, k1=100, k2=30, lambda_value=0.3)
                dist_1 = build_dist(query_feats_2, gallery_feats, "euclidean")
                rerank_dist = build_dist(query_feats_2, gallery_feats, metric="jaccard", k1=100, k2=30)
                dist = rerank_dist * (1 - 0.3) + dist_1 * 0.3
                sim_2 = -dist
            else:
                sim_2 = query_feats_2 @ gallery_feats.t()
            sim += sim_2
            q_num += 1

        if query_feats_3 is not None:
            query_feats_3 = F.normalize(query_feats_3.float(), p=2, dim=1)
            if self.args.rerank:
                # sim_3 = -re_ranking(query_feats_3, gallery_feats, k1=100, k2=30, lambda_value=0.3)
                dist_1 = build_dist(query_feats_3, gallery_feats, "euclidean")
                rerank_dist = build_dist(query_feats_3, gallery_feats, metric="jaccard", k1=100, k2=30)
                dist = rerank_dist * (1 - 0.3) + dist_1 * 0.3
                sim_3 = -dist
            else:
                sim_3 = query_feats_3 @ gallery_feats.t()
            sim += sim_3
            q_num += 1

        if query_feats_4 is not None:
            query_feats_4 = F.normalize(query_feats_4.float(), p=2, dim=1)
            if self.args.rerank:
                # sim_4 = -re_ranking(query_feats_4, gallery_feats, k1=100, k2=30, lambda_value=0.3)
                dist_1 = build_dist(query_feats_4, gallery_feats, "euclidean")
                rerank_dist = build_dist(query_feats_4, gallery_feats, metric="jaccard", k1=100, k2=30)
                dist = rerank_dist * (1 - 0.3) + dist_1 * 0.3
                sim_4 = -dist
            else:
                sim_4 = query_feats_4 @ gallery_feats.t()
            sim += sim_4
            q_num += 1
        sim = sim / q_num

        # 测试
        if mode == 'test':
            sim = torch.tensor(sim).to(query_feats.device)
            # _,indices = torch.topk(sim,k=100,dim=1,largest=True,sorted=True)  #### 真实测试集不再取 前 100, 而是所有 gallery
            _, indices = torch.sort(sim, dim=1, descending=True)  # 返回所有gallery结果

            self.all_query_ids_list.extend(query_ids.tolist())
            self.all_gallery_ids_list.extend((indices.cpu() + 1).int().tolist())
            # for r in tqdm(range(indices.shape[0])):
                # self.all_query_ids_list.append(int(query_ids[r]))
                # self.all_gallery_ids_list.append(list(map(int, indices[r].cpu() + 1)))  # indices是从0开始的,gallery最后是从1.jpg开始
            self.logger.info('finish one query mode')
            return None
        
        if not self.args.rerank:
            sim = sim.detach().cpu()#.numpy()

        # t2i_cmc, t2i_mAP, t2i_mINP = eval_func(
        #     -sim, query_ids.numpy(), gallery_ids.numpy(),
        #     query_image_ids.numpy(), gallery_image_ids.numpy(), set=0, max_rank=10)
        
        # 去掉 query image ids, gallery image ids
        t2i_cmc, t2i_mAP, t2i_mINP, _ = rank(
            similarity=sim,
            q_pids=query_ids,
            g_pids=gallery_ids,
            max_rank=10,
            get_mAP=True
        )

        return t2i_cmc[0], t2i_cmc[4], t2i_cmc[9], t2i_mAP, t2i_mINP
    
    def _fourmodal_iiit(self, img_img_img_text_loader, device, model, input_img_type1=None, input_img_type2=None, input_img_type3=None):
        """
        输入是 cp, sk, nir, text,
        """
        qids,qfeats_text,qfeats_text_img_img_img, qimage_ids, qimage_ids2=[],[],[],[],[]
        qfeats_img, qfeats_img2, qfeats_img3 = [], [], []
        for pid, img, img2, img3, caption in img_img_img_text_loader:
            caption = caption.to(device)
            img = img.to(device)
            img2 = img2.to(device)
            img3 = img3.to(device)
            with torch.no_grad():
                text_feat = model.encode_text(caption)
                img_feat = model.encode_image(img, input_img_type1)  # cp
                img2_feat = model.encode_image(img2, input_img_type2) # sk
                img3_feat = model.encode_image(img3, input_img_type3)  # nir

                # img_feat_flip = model.encode_image(img.flip(3), input_img_type1)  # cp
                # img2_feat_flip = model.encode_image(img2.flip(3), input_img_type2) # sk
                # img3_feat_flip = model.encode_image(img3.flip(3), input_img_type3)  # nir

                if self.args.fusion_way in ['add', 'weight add', 'cross attention', 'parameter add', 'concat', 'global concat', 'cross attention text', 'cross attention sketch', 'concat transformer']:
                    text_img_img_img_fu = model.fusion_layer(
                        text_feat, img2_feat, img_feat, img3_feat, caption, way=self.args.fusion_way  ### 注意顺序, 数据是 cp, sk ,nir, 但是fusion layer是 txt, sk ,cp , nir
                        )
                    # img_feat = (img_feat[:, 0, :].float() + img_feat_flip[:, 0, :].float()) / 2
                    # img2_feat = (img2_feat[:, 0, :].float() + img2_feat_flip[:, 0, :].float()) / 2
                    # img3_feat = (img3_feat[:, 0, :].float() + img3_feat_flip[:, 0, :].float()) / 2
                    img_feat = img_feat[:, 0, :].float()
                    img2_feat = img2_feat[:, 0, :].float()
                    img3_feat = img3_feat[:, 0, :].float()

                text_feat = text_feat[torch.arange(text_feat.shape[0]), caption.argmax(dim=-1)].float()
            qids.append(pid.view(-1).cpu()) # flatten 
            qfeats_text_img_img_img.append(text_img_img_img_fu.cpu())
            # qimage_ids.append(image_id)

            qfeats_text.append(text_feat.cpu())
            qfeats_img.append(img_feat.cpu())
            qfeats_img2.append(img2_feat.cpu())
            qfeats_img3.append(img3_feat.cpu())

        qids = torch.cat(qids, 0)
        qfeats_text_img_img_img = torch.cat(qfeats_text_img_img_img, 0)
        # qimage_ids = torch.cat(qimage_ids, 0)
        
        qfeats_text = torch.cat(qfeats_text, 0)
        qfeats_img = torch.cat(qfeats_img, 0)
        qfeats_img2 = torch.cat(qfeats_img2, 0)
        qfeats_img3 = torch.cat(qfeats_img3, 0)
        return qids, qfeats_text_img_img_img, qfeats_img, qfeats_img2, qfeats_img3, qfeats_text
    
    def _threemodal_iii(self, img_img_img_loader, device, model, input_img_type1=None, input_img_type2=None, input_img_type3=None):
        qids, qfeats_img_img_img, qimage_ids=[],[],[]
        qfeats_img, qfeats_img2, qfeats_img3 = [], [], []
        for pid, img, img2, img3 in img_img_img_loader:
            img3 = img3.to(device)
            img2 = img2.to(device)
            img = img.to(device)
            with torch.no_grad():
                img3_feat = model.encode_image(img3, input_img_type3)
                img2_feat = model.encode_image(img2, input_img_type2)
                img_feat = model.encode_image(img, input_img_type1)

                # img3_feat_flip = model.encode_image(img3.flip(3), input_img_type3)
                # img2_feat_flip = model.encode_image(img2.flip(3), input_img_type2)
                # img_feat_flip = model.encode_image(img.flip(3), input_img_type1)

                if self.args.fusion_way in ['add', 'weight add', 'cross attention', 'parameter add', 'concat', 
                                            'global concat', 'cross attention text', 'cross attention sketch', 'concat transformer']:
                    img_img_img_fu = model.fusion_layer(
                        None, img2_feat, img_feat, img3_feat, None, way=self.args.fusion_way  ### 注意顺序, 数据是 cp, sk ,nir, 但是fusion layer是 txt, sk ,cp , nir
                        )
                    # img_feat = (img_feat[:, 0, :].float() + img_feat_flip[:, 0, :].float()) / 2
                    # img2_feat = (img2_feat[:, 0, :].float() + img2_feat_flip[:, 0, :].float()) / 2
                    # img3_feat = (img3_feat[:, 0, :].float() + img3_feat_flip[:, 0, :].float()) / 2
                    img_feat = img_feat[:, 0, :].float()
                    img2_feat = img2_feat[:, 0, :].float()
                    img3_feat = img3_feat[:, 0, :].float()
                
            qids.append(pid.view(-1).cpu())
            qfeats_img_img_img.append(img_img_img_fu.cpu())
            # qimage_ids.append(image_id)
            qfeats_img.append(img_feat.cpu())
            qfeats_img2.append(img2_feat.cpu())
            qfeats_img3.append(img3_feat.cpu())
            

        qids = torch.cat(qids, 0)
        qfeats_img_img_img = torch.cat(qfeats_img_img_img, 0)

        # qimage_ids = torch.cat(qimage_ids, 0)
        
        qfeats_img = torch.cat(qfeats_img, 0)
        qfeats_img2 = torch.cat(qfeats_img2, 0)
        qfeats_img3 = torch.cat(qfeats_img3, 0)

        return qids, qfeats_img_img_img, qfeats_img, qfeats_img2, qfeats_img3
    
    def _threemodal_iit(self, img_img_text_loader, device, model, input_img_type1=None, input_img_type2=None):
        qids,qfeats_text,qfeats_text_img_img, qimage_ids, qimage_ids2=[],[],[],[],[]
        qfeats_img, qfeats_img2 = [], []
        for pid, img, img2, caption in img_img_text_loader:
            caption = caption.to(device)
            img = img.to(device)
            img2 = img2.to(device)
            with torch.no_grad():
                text_feat = model.encode_text(caption)
                img_feat = model.encode_image(img, input_img_type1)
                img2_feat = model.encode_image(img2, input_img_type2)

                # img_feat_flip = model.encode_image(img.flip(3), input_img_type1)
                # img2_feat_flip = model.encode_image(img2.flip(3), input_img_type2)

                if self.args.fusion_way in ['add', 'weight add', 'cross attention', 
                                            'parameter add', 'concat', 'global concat', 'cross attention text', 'cross attention sketch', 'concat transformer']:
                    if input_img_type1 == InputImageType.color_pencil and input_img_type2 == InputImageType.sketch:
                        text_img_img_fu = model.fusion_layer(
                            text_feat, img2_feat, img_feat, None, caption, way=self.args.fusion_way  ### 注意顺序: text, sk, cp, nir
                            )
                        img_feat = img_feat[:, 0, :].float()
                        # img_feat = (img_feat + img_feat_flip[:, 0, :].float()) / 2

                        img2_feat = img2_feat[:, 0, :].float()
                        # img2_feat = (img2_feat + img2_feat_flip[:, 0, :].float()) / 2
                    elif input_img_type1 == InputImageType.sketch and input_img_type2 == InputImageType.nir:
                        text_img_img_fu = model.fusion_layer(
                            text_feat, img_feat, None, img2_feat, caption, way=self.args.fusion_way
                            )
                        img_feat = img_feat[:, 0, :].float()
                        # img_feat = (img_feat + img_feat_flip[:, 0, :].float()) / 2

                        img2_feat = img2_feat[:, 0, :].float()
                        # img2_feat = (img2_feat + img2_feat_flip[:, 0, :].float()) / 2
                    elif input_img_type1 == InputImageType.color_pencil and input_img_type2 == InputImageType.nir:
                        text_img_img_fu = model.fusion_layer(
                            text_feat, None, img_feat, img2_feat, caption, way=self.args.fusion_way
                            )
                        img_feat = img_feat[:, 0, :].float()
                        # img_feat = (img_feat + img_feat_flip[:, 0, :].float()) / 2

                        img2_feat = img2_feat[:, 0, :].float()
                        # img2_feat = (img2_feat + img2_feat_flip[:, 0, :].float()) / 2

                    text_feat = text_feat[torch.arange(text_feat.shape[0]), caption.argmax(dim=-1)].float()

                
            qids.append(pid.view(-1).cpu()) # flatten 
            # qfeats_text.append(text_feat)
            qfeats_text_img_img.append(text_img_img_fu.cpu())
            # qfeats_text_img_img_tse.append(text_img_img_fu_tse)

            # qimage_ids.append(image_id)
            # qimage_ids2.append(image_id2)
            qfeats_img.append(img_feat.cpu())#[:,0,:].float())
            qfeats_img2.append(img2_feat.cpu())#[:,0,:].float())
            qfeats_text.append(text_feat.cpu())#[torch.arange(text_feat.shape[0]), caption.argmax(dim=-1)].float())

        qids = torch.cat(qids, 0)
        qfeats_text_img_img = torch.cat(qfeats_text_img_img, 0)
        # qfeats_text_img_img_tse = torch.cat(qfeats_text_img_img_tse, 0)
        # qimage_ids = torch.cat(qimage_ids, 0)

        qfeats_text = torch.cat(qfeats_text, 0)
        qfeats_img = torch.cat(qfeats_img, 0)
        qfeats_img2 = torch.cat(qfeats_img2, 0)

        
        return qids, qfeats_text_img_img, qfeats_img, qfeats_img2, qfeats_text#, \
                # qfeats_img_tse,qfeats_img2_tse, qfeats_text_tse, qfeats_text_img_img_tse
    
    def _twomodal_it(self, img_text_loader, device, model, input_img_type=None):
        qids,qfeats_text_img, qimage_ids=[],[],[]
        qfeats_img, qfeats_text = [], []

        # qfeats_img_tse, qfeats_text_tse = [], []
        # qfeats_text_img_tse = []
        for pid, img, caption in img_text_loader:
            caption = caption.to(device)
            img = img.to(device)
            with torch.no_grad():
                text_feat = model.encode_text(caption)
                img_feat = model.encode_image(img, input_img_type)
                # img_feat_flip = model.encode_image(img.flip(3), input_img_type)

                if self.args.fusion_way in ['add', 'weight add', 'cross attention', 
                                            'parameter add', 'concat', 'global concat', 'cross attention text', 'cross attention sketch', 'concat transformer']:
                    if input_img_type == InputImageType.sketch:
                        text_img_text_fu = model.fusion_layer(
                            text_feat, img_feat, None, None, caption, way=self.args.fusion_way)
                        # img_feat = (img_feat[:, 0, :].float() + img_feat_flip[:, 0, :].float()) / 2

                    elif input_img_type == InputImageType.color_pencil:
                        text_img_text_fu = model.fusion_layer(
                            text_feat, None, img_feat, None, caption, way=self.args.fusion_way)
                        # img_feat = (img_feat[:, 0, :].float() + img_feat_flip[:, 0, :].float()) / 2

                    elif input_img_type == InputImageType.nir:
                        text_img_text_fu = model.fusion_layer(
                            text_feat, None, None, img_feat, caption, way=self.args.fusion_way)
                        # img_feat = (img_feat[:, 0, :].float() + img_feat_flip[:, 0, :].float()) / 2

                    text_feat = text_feat[torch.arange(text_feat.shape[0]), caption.argmax(dim=-1)].float()
                    img_feat = img_feat[:, 0, :].float()

                
            qids.append(pid.view(-1).cpu()) # flatten 
            qfeats_text_img.append(text_img_text_fu.cpu())
            # qimage_ids.append(image_id)

            qfeats_img.append(img_feat.cpu())#[:,0,:].float())
            qfeats_text.append(text_feat.cpu())#[torch.arange(text_feat.shape[0]), caption.argmax(dim=-1)].float())

        qids = torch.cat(qids, 0)
        qfeats_text_img = torch.cat(qfeats_text_img, 0)
        # qimage_ids = torch.cat(qimage_ids, 0)

        qfeats_img = torch.cat(qfeats_img, 0)
        qfeats_text = torch.cat(qfeats_text, 0)

        return qids, qfeats_text_img, qfeats_img, qfeats_text#, qfeats_img_tse, qfeats_text_tse, qfeats_text_img_tse
    

    def _twomodal_ii(self, img_img_loader, device, model, input_img_type1=None, input_img_type2=None):
        qids, qfeats_img_img, qimage_ids=[],[],[]
        qfeats_img, qfeats_img2 = [], []

        # qfeats_img_tse, qfeats_img2_tse = [], []
        # qfeats_img_img_tse = []
        for pid, img, img2 in img_img_loader:
            # assert pid == pid2
            img2 = img2.to(device)
            img = img.to(device)
            with torch.no_grad():
                img2_feat = model.encode_image(img2, input_img_type2)
                img_feat = model.encode_image(img, input_img_type1)

                # img2_feat_flip = model.encode_image(img2.flip(3), input_img_type2)
                # img_feat_flip = model.encode_image(img.flip(3), input_img_type1)

                # img2_feat_tse = model.encode_image_tse(img2, input_img_type2)
                # img_feat_tse = model.encode_image_tse(img, input_img_type1)

                if self.args.fusion_way in ['add', 'weight add', 'cross attention', 
                                            'parameter add', 'concat', 'global concat', 'cross attention text', 'cross attention sketch', 'concat transformer']:
                    if input_img_type1 == InputImageType.sketch and input_img_type2 == InputImageType.color_pencil:
                        img_img_fu = model.fusion_layer(
                            None, img_feat, img2_feat, None, None, way=self.args.fusion_way)
                        img_feat = img_feat[:, 0, :].float()
                        # img_feat = (img_feat + img_feat_flip[:, 0, :].float()) / 2

                        img2_feat = img2_feat[:, 0, :].float()
                        # img2_feat = (img2_feat + img2_feat_flip[:, 0, :].float()) / 2

                    elif input_img_type1 == InputImageType.sketch and input_img_type2 == InputImageType.nir:
                        img_img_fu = model.fusion_layer(
                            None, img_feat, None, img2_feat, None, way=self.args.fusion_way)
                        img_feat = img_feat[:, 0, :].float()
                        # img_feat = (img_feat + img_feat_flip[:, 0, :].float()) / 2

                        img2_feat = img2_feat[:, 0, :].float()
                        # img2_feat = (img2_feat + img2_feat_flip[:, 0, :].float()) / 2

                    elif input_img_type1 == InputImageType.color_pencil and input_img_type2 == InputImageType.nir:
                        img_img_fu = model.fusion_layer(
                            None, None, img_feat, img2_feat, None, way=self.args.fusion_way)
                        img_feat = img_feat[:, 0, :].float()
                        # img_feat = (img_feat + img_feat_flip[:, 0, :].float()) / 2

                        img2_feat = img2_feat[:, 0, :].float()
                        # img2_feat = (img2_feat + img2_feat_flip[:, 0, :].float()) / 2
                
            qids.append(pid.view(-1).cpu()) # flatten 
            # qfeats_img.append(img_feat)
            qfeats_img_img.append(img_img_fu.cpu())
            # qfeats_img_img_tse.append(img_img_fu_tse)
            # qimage_ids.append(image_id)

            qfeats_img.append(img_feat.cpu())#[:,0,:].float())
            qfeats_img2.append(img2_feat.cpu())#[:,0,:].float())

            

        qids = torch.cat(qids, 0)
        # qfeats_text = torch.cat(qfeats_text, 0)
        qfeats_img_img = torch.cat(qfeats_img_img, 0)
        # qfeats_img_img_tse = torch.cat(qfeats_img_img_tse, 0)
        # qimage_ids = torch.cat(qimage_ids, 0)

        qfeats_img = torch.cat(qfeats_img, 0)
        qfeats_img2 = torch.cat(qfeats_img2, 0)
        # qfeats_img_tse = torch.cat(qfeats_img_tse, 0)
        # qfeats_img2_tse = torch.cat(qfeats_img2_tse, 0)
        return qids, qfeats_img_img, qfeats_img, qfeats_img2#, qfeats_img_tse, qfeats_img2_tse, qfeats_img_img_tse


    def _onemodal(self, data_loader, device, model, input_img_type=None):
        qids, qfeats_img, qimage_ids=[],[],[]
        # qfeats_img_tse = []

        for pid, img in data_loader:
            img = img.to(device)
            with torch.no_grad():
                # img_feat = model.encode_image(img, input_img_type)[:, 0, :].float()  ### 全局
                img_feat = model.encode_image(img, input_img_type)

                # flip enabled for testing augmentation ? 
                # img_feat_flip = model.encode_image(img.flip(3), input_img_type)
                if input_img_type == InputImageType.rgb:
                    img_feat = img_feat[:, 0, :].float()
                    # img_feat = ( img_feat + img_feat_flip[:, 0, :].float() ) / 2
                    
                elif input_img_type == InputImageType.sketch:
                    img_feat = img_feat[:, 0, :].float()
                    # img_feat = ( img_feat + img_feat_flip[:, 0, :].float() ) / 2

                elif input_img_type == InputImageType.nir:
                    img_feat = img_feat[:, 0, :].float()
                    # img_feat = ( img_feat + img_feat_flip[:, 0, :].float() ) / 2

                elif input_img_type == InputImageType.color_pencil:
                    img_feat = img_feat[:, 0, :].float()
                    # img_feat = ( img_feat + img_feat_flip[:, 0, :].float() ) / 2

            qids.append(pid.view(-1).cpu()) # flatten
            qfeats_img.append(img_feat.cpu())

            # qfeats_img_tse.append(img_feat_tse)
            # qimage_ids.append(image_id)

        qids = torch.cat(qids, 0)
        qfeats_img = torch.cat(qfeats_img, 0)
        # qfeats_img_tse = torch.cat(qfeats_img_tse, 0)
        # qimage_ids = torch.cat(qimage_ids, 0)
        return qids, qfeats_img#, qimage_ids#, qfeats_img_tse


    def _onemodal_t(self, data_loader, device, model):
        qids, qfeats_text, qtext_ids=[],[],[]
        # qfeats_text_tse = []
        for pid, caption in data_loader:
            caption = caption.to(device)
            with torch.no_grad():
                text_feat = model.encode_text(caption)
                # text_feat_tse = model.encode_text_tse(caption)

                # text_feat = text_feat[torch.arange(text_feat.shape[0]), caption.argmax(dim=-1)].float()
                text_feat = text_feat[torch.arange(text_feat.shape[0]), caption.argmax(dim=-1)].float()
            qids.append(pid.view(-1).cpu()) # flatten
            qfeats_text.append(text_feat.cpu()) 
            # tse embedding
            # qfeats_text_tse.append(text_feat_tse)

            # qtext_ids.append(caption_id) 

        qids = torch.cat(qids, 0)
        qfeats_text = torch.cat(qfeats_text, 0)
        # qfeats_text_tse = torch.cat(qfeats_text_tse, 0)
        # qtext_ids = torch.cat(qtext_ids, 0)

        return qids, qfeats_text#, qtext_ids#, qfeats_text_tse

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
    