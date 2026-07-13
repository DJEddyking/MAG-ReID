from prettytable import PrettyTable
import torch
import torch.nn.functional as F
import logging
from typing import List, Tuple, Callable, Dict, Any
from model.utils import InputImageType
from utils.simple_tokenizer import SimpleTokenizer
from datasets.bases import tokenize
import random
import copy
import numpy as np
from utils.runtime_metrics import QueryLatencyMeter, measure_forward_latency


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

    return all_cmc, mAP, mINP, indices

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

# def eval_func(distmat, q_pids, g_pids, q_camids, g_camids, max_rank=50):
#     """Evaluation with market1501 metric
#         Key: for each query identity, its gallery images from the same camera view are discarded.
#         """
#     num_q, num_g = distmat.shape
#     # distmat g
#     #    q    1 3 2 4
#     #         4 1 2 3
#     if num_g < max_rank:
#         max_rank = num_g
#         print("Note: number of gallery samples is quite small, got {}".format(num_g))
#     indices = np.argsort(distmat, axis=1)
#     #  0 2 1 3
#     #  1 2 3 0
#     matches = (g_pids[indices] == q_pids[:, np.newaxis]).astype(np.int32)
#     # compute cmc curve for each query
#     all_cmc = []
#     all_AP = []
#     num_valid_q = 0.  # number of valid query
#     for q_idx in range(num_q):
#         # get query pid and camid
#         q_pid = q_pids[q_idx]
#         q_camid = q_camids[q_idx]

#         # remove gallery samples that have the same pid and camid with query
#         order = indices[q_idx]  # select one row
#         remove = (g_pids[order] == q_pid) & (g_camids[order] == q_camid)
#         keep = np.invert(remove)

#         # compute cmc curve
#         # binary vector, positions with value 1 are correct matches
#         orig_cmc = matches[q_idx][keep]
#         if not np.any(orig_cmc):
#             # this condition is true when query identity does not appear in gallery
#             continue

#         cmc = orig_cmc.cumsum()
#         cmc[cmc > 1] = 1

#         all_cmc.append(cmc[:max_rank])
#         num_valid_q += 1.

#         # compute average precision
#         # reference: https://en.wikipedia.org/wiki/Evaluation_measures_(information_retrieval)#Average_precision
#         num_rel = orig_cmc.sum()
#         tmp_cmc = orig_cmc.cumsum()
#         # tmp_cmc = [x / (i + 1.) for i, x in enumerate(tmp_cmc)]
#         y = np.arange(1, tmp_cmc.shape[0] + 1) * 1.0
#         tmp_cmc = tmp_cmc / y
#         tmp_cmc = np.asarray(tmp_cmc) * orig_cmc
#         AP = tmp_cmc.sum() / num_rel
#         all_AP.append(AP)

#     assert num_valid_q > 0, "Error: all query identities do not appear in gallery"

#     all_cmc = np.asarray(all_cmc).astype(np.float32)
#     all_cmc = all_cmc.sum(0) / num_valid_q
#     mAP = np.mean(all_AP)

#     return all_cmc * 100, mAP * 100

class Evaluator():
    def __init__(self, args, gallery_loader, get_mAP=True, num_query=None, **query_loaders):
        self.args = args
        self.gallery_loader = gallery_loader
        self.query_loaders = query_loaders
        self.get_mAP = get_mAP
        self.num_query = num_query
        self.logger = logging.getLogger("ORBench.eval")
        # RGBNT201/RGBNT201_Text 评测模式：
        # - "concat"：RGB+NI+TI 特征拼接后做一次检索（默认，保持当前行为）
        # - "per_modality_avg"：分别用 RGB↔RGB、NI↔NI、TI↔TI 计算指标，再对指标取平均
        # self.rnt_eval_mode = getattr(self.args, "rnt_eval_mode", "concat")

        self.tokenizer = SimpleTokenizer()
        # self.rgb_caption_prefix = "Visible image of a" + " X X " + "person with natural colors: " # 4 个, 不加sos token; rgb不加多的提示词? TODO 
        # model.nir_caption_prefix = "Near-infrared image of a" + " X X " + "person with high reflectance contrast: " # 6 个
        # model.cp_caption_prefix = "Color-pencil drawing of a" + " X X " + "person with vivid colors: " # 6个
        # model.sk_caption_prefix = "Sketch image of a" + " X X " + "person with clean line contours: "  ## 4个
        # self.text_length = self.args.text_length

    def _input_img_type(self, modality):
        if modality == 'sk':
            return InputImageType.sketch
        if modality == 'cp':
            return InputImageType.color_pencil
        if modality == 'nir':
            return InputImageType.nir
        if modality == 'rgb':
            return InputImageType.rgb
        raise ValueError("Unsupported image modality: {}".format(modality))

    def _text_feature_from_tokens(self, text_seq, token_ids):
        eos_index = token_ids.argmax(dim=-1)
        batch_index = torch.arange(text_seq.shape[0], device=token_ids.device)
        return text_seq[batch_index, eos_index].float()

    def _forward_multi_modality_query(self, model, imgs, modalities):
        if len(modalities) == 1:
            if modalities[0] == 'text':
                text_seq = model.encode_text(imgs[0], modality=InputImageType.rgb, use_prompt=False)
                return [self._text_feature_from_tokens(text_seq, imgs[0])]
            img_feat, _ = model.encode_image(imgs[0], self._input_img_type(modalities[0]))
            return [img_feat]

        if len(modalities) == 2 and 'text' in modalities:
            caption = imgs[-1]
            visual_modality = [modality for modality in modalities if modality != 'text'][0]
            text_seq = model.encode_text(caption, use_prompt=False)
            text_feat = self._text_feature_from_tokens(text_seq, caption)
            img_feat, _ = model.encode_image(imgs[0], self._input_img_type(visual_modality))
            if self.args.new_eval:
                img_feat, text_feat_use_inverse = model.encode_image(
                    imgs[0],
                    self._input_img_type(visual_modality),
                    model.inverseNet(text_feat).half(),
                )
                text_feat = model._transformer(text_feat + text_feat_use_inverse)
            return [img_feat, text_feat]

        if 'text' in modalities:
            caption = imgs[-1]
            visual_modalities = [modality for modality in modalities if modality != 'text']
            text_seq = model.encode_text(caption, use_prompt=False)
            feats = [self._text_feature_from_tokens(text_seq, caption)]
            for img, modality in zip(imgs[:-1], visual_modalities):
                img_feat, _ = model.encode_image(img, self._input_img_type(modality))
                feats.append(img_feat)
            return feats

        feats = []
        for img, modality in zip(imgs, modalities):
            img_feat, _ = model.encode_image(img, self._input_img_type(modality))
            feats.append(img_feat)
        return feats

    def _extract_multi_modality_features_with_forward_latency(self, model, loader, modalities, task_name):
        """Extract query features and measure only model forward latency in milliseconds."""
        model = model.eval()
        device = next(model.parameters()).device
        qids, qimage_ids = [], []
        feature_groups = []

        for batch in loader:
            pid = batch[0]
            if self.args.dataset_name in ['CUHK-PEDES', 'ICFG-PEDES', 'RSTPReid', 'PKU-Sketch'] and 'sk' in modalities:
                image_ids = batch[-1]
                qimage_ids.append(image_ids.cpu())

            imgs = [img.to(device) for img in batch[1:1 + len(modalities)]]

            with torch.no_grad():
                feats, elapsed_ms = measure_forward_latency(
                    lambda: self._forward_multi_modality_query(model, imgs, modalities),
                    device,
                )

            latency_meter = getattr(self, "query_latency_meter", None)
            if latency_meter is not None:
                batch_size = pid.numel() if hasattr(pid, "numel") else len(pid)
                latency_meter.update(task_name, len(modalities), elapsed_ms, batch_size)

            if not feature_groups:
                feature_groups = [[] for _ in feats]
            for idx, feat in enumerate(feats):
                feature_groups[idx].append(feat.cpu())
            qids.append(pid.view(-1).cpu())

        qids = torch.cat(qids, 0)
        qfeats = [F.normalize(torch.cat(feats, 0), p=2, dim=1) for feats in feature_groups]
        qfeats += [None] * (6 - len(qfeats))
        return qids, qfeats[0], qfeats[1], qfeats[2], qfeats[3], qfeats[4], qfeats[5], qimage_ids

    def _extract_multi_modality_features(self, model, loader, modalities):
        """Extract fused features for multiple modalities"""
        model = model.eval()
        device = next(model.parameters()).device
        qids, qfeats = [], []

        embeds = []
        embeds_1, embeds_2, embeds_3, embeds_4, embeds_5 = [], [], [], [], []
        qimage_ids = []

        # embeds_t = []
        # embeds_1_t, embeds_2_t, embeds_3_t, embeds_4_t = [], [], [], []  ## 局部特征

        # Determine the number of images based on modalities count
        num_modalities = len(modalities)

        # mask_token = self.tokenizer.encoder["<|mask|>"]

        for batch in loader:
            pid = batch[0]
            # 对于tri cuhk数据集, 有图像的, 最后一个是image id !!!
            # 只有sk的时候才需要image ids, text不需要
            if self.args.dataset_name in ['CUHK-PEDES', 'ICFG-PEDES', 'RSTPReid', 'PKU-Sketch'] and 'sk' in modalities:
                image_ids = batch[-1]
                qimage_ids.append(image_ids.cpu())

            imgs = [img.to(device) for img in batch[1:1 + num_modalities]]  # 对于2个模态以上的, 这个可能会有caption

            with torch.no_grad():
                # 加上text: 可以单独提出来
                # sk
                # txt_prefix = tokenize(model.sk_caption_prefix, tokenizer=self.tokenizer, text_length=self.text_length, truncate=True).to(imgs[0].device)
                # text_feat_sk = torch.mean(model.encode_text(txt_prefix.repeat(pid.shape[0], 1), modality=InputImageType.sketch, use_prompt=True)[:, 5: 5 + 2], dim=1)
                # text_feat_sk = model._use_inverse(txt_prefix.repeat(pid.shape[0], 1),  )
                # 增加inverse
                # text_feat_sk = model.inverseNet(text_feat_sk).float()
                # cp
                # txt_prefix = tokenize(model.cp_caption_prefix, tokenizer=self.tokenizer, text_length=self.text_length, truncate=True).to(imgs[0].device)
                # text_feat_cp = torch.mean(model.encode_text(txt_prefix.repeat(pid.shape[0], 1), modality=InputImageType.color_pencil, use_prompt=True)[:, 7: 7 + 2], dim=1)
                # text_feat_cp = model.inverseNet(text_feat_cp).float()
                # nir
                # txt_prefix = tokenize(model.nir_caption_prefix, tokenizer=self.tokenizer, text_length=self.text_length, truncate=True).to(imgs[0].device)
                # text_feat_nir = torch.mean(model.encode_text(txt_prefix.repeat(pid.shape[0], 1), modality=InputImageType.nir, use_prompt=True)[:, 7: 7 + 2], dim=1)
                # text_feat_nir = model.inverseNet(text_feat_nir).float()

                # 单模态
                if len(modalities) == 1:
                    if modalities[0] == 'text':
                        # 先decoder, 然后加上prompt, 然后再encode
                        # text_feat = self._get_new_text_feat_with_only_text(imgs[0], model)
                        # 只有text的不操作
                        text_feat = model.encode_text(imgs[0], modality=InputImageType.rgb, use_prompt=False)
                        text_feat = text_feat[torch.arange(text_feat.shape[0]), imgs[0].argmax(dim=-1)].float()

                        # fuse_feat = model._transformer_shared(text_feat.unsqueeze(1).permute(1, 0, 2)).permute(1, 0, 2).mean(dim=1)
                        # embeds_1.append(fuse_feat.cpu())
                        # if self.args.new_eval and self.args.use_local_feat:
                        #     # 把局部特征也加进去评测, 对于text随机mask
                        #     masked_caption_ids = self._build_random_masked_tokens_and_labels(copy.deepcopy(imgs[0]).cpu().numpy()).to(imgs[0].device)
                        #     text_feat_mask = model.encode_text(masked_caption_ids, modality=InputImageType.rgb, use_prompt=False)
                        #     masked_position_idx_rgb = model._get_mask_position_lst(masked_caption_ids, mask_token)
                        #     masked_feats = []
                        #     for xx in range(len(masked_position_idx_rgb)):
                        #         masked_feats.append(text_feat_mask[xx][masked_position_idx_rgb[xx], :].mean(dim=0).float())
                        #     masked_feats = torch.stack(masked_feats, dim=0)
                        #     embeds_1.append(masked_feats.cpu())

                        embeds.append(text_feat.cpu())

                        
                    else:
                        """
                        提升单模态能力: 1.引入TTA ?? 比如对图像采用数据增强, 然后以熵作为权重将特征相加? 或者直接相加特征 
                        2.直接把局部的特征拿过来一起做检索: 训练的时候用上局部特征与全局约束, 推理经过text encoder之后辅助检索
                        3. modal token
                        4. fusion feature
                        """
                        input_img_type = None
                        if modalities[0] == 'sk':
                            input_img_type = InputImageType.sketch
                            # prefix_txt_token = tokenize(model.sk_caption_prefix, tokenizer=self.tokenizer, text_length=self.args.text_length, truncate=True).to(imgs[0].device)

                        elif modalities[0] == 'cp':
                            input_img_type = InputImageType.color_pencil
                            # prefix_txt_token = tokenize(model.cp_caption_prefix, tokenizer=self.tokenizer, text_length=self.args.text_length, truncate=True).to(imgs[0].device)

                        elif modalities[0] == 'nir':
                            input_img_type = InputImageType.nir
                            # prefix_txt_token = tokenize(model.nir_caption_prefix, tokenizer=self.tokenizer, text_length=self.args.text_length, truncate=True).to(imgs[0].device)

                        img_feat, _ = model.encode_image(imgs[0], input_img_type)
                        # fusion_feat = model._transformer_shared(img_feat.unsqueeze(1).permute(1,0,2)).permute(1,0,2).mean(dim=1)
                        # _, _, image_features_proj = model.base_model.encode_image(imgs[0], input_img_type)
                        # img_feat = image_features_proj[:, 0, :].float()

                        # 单模态用学到的prompt增强 ?
                        # _, _, image_features_proj, txt_global = model._use_inverse(prefix_txt_token.repeat(imgs[0].shape[0], 1),  imgs[0], input_img_type, use_prompt=True)
                        # img_feat, text_feat = image_features_proj[:, 0, :].float(), image_features_proj[:, -1, :].float()
                        
                        embeds.append(img_feat.cpu())

                        # if self.args.use_local_feat:
                            # img_feat_local = model.conv_inverse_net(image_features_proj[:, 1:, :].float())
                            # embeds_1.append(img_feat_local.cpu())
                        
                        # img_feat_t = model.encode_image_tse(imgs[0], input_img_type)
                        # embeds_t.append(img_feat_t.cpu())
                        # embeds_1.append(fusion_feat.cpu())
                        # embeds_2.append(txt_global.cpu())

                elif len(modalities) == 2:
                    if 'text' in modalities:
                        # caption在base.py中统一放到了最后面
                        # 对于双模态, text和image使用混合后的; 同时保证原始的text

                        ## 只要text 在里面, 就用rgb new caption
                        # text_feat = self._get_new_text_feat_with_only_text(imgs[-1], model)

                        text_feat_origin = model.encode_text(imgs[-1], use_prompt=False)
                        text_feat_origin = text_feat_origin[torch.arange(text_feat_origin.shape[0]), imgs[-1].argmax(dim=-1)].float()

                        input_img_type = None
                        if modalities[1] == 'sk':
                            input_img_type = InputImageType.sketch
                            # 拼接前缀修改text
                            # imgs[-1] = torch.stack(self._get_new_token(imgs[-1].cpu().tolist(), model.sk_caption_prefix), dim=0).to(imgs[0].device)

                        elif modalities[1] == 'cp':
                            input_img_type = InputImageType.color_pencil
                            # imgs[-1] = torch.stack(self._get_new_token(imgs[-1].cpu().tolist(), model.cp_caption_prefix), dim=0).to(imgs[0].device)

                        elif modalities[1] == 'nir':
                            input_img_type = InputImageType.nir
                            # imgs[-1] = torch.stack(self._get_new_token(imgs[-1].cpu().tolist(), model.nir_caption_prefix), dim=0).to(imgs[0].device)

                        img_feat, _ = model.encode_image(imgs[0], input_img_type)
                        # fusion_feat = model._transformer_img_shared(img_feat)
                        # _, _, image_features_proj, txt_global = model._use_inverse(imgs[-1],  imgs[0], input_img_type, use_prompt=True)  #### 
                        # img_feat, text_feat = image_features_proj[:, 0, :].float(), image_features_proj[:, -1, :].float()
                        # text_feat = text_feat + txt_global

                        if self.args.new_eval:
                            img_feat, text_feat_use_inverse = model.encode_image(imgs[0], input_img_type, model.inverseNet(text_feat_origin).half())
                            text_feat_origin = model._transformer(text_feat_origin + text_feat_use_inverse)
                        # 融合模态: text ,sk ,cp ,nir
                        # if input_img_type == InputImageType.sketch:
                        #     text_img_text_fu = model.fusion_layer(
                        #         text_feat, img_feat, None, None, imgs[-1], way=self.args.fusion_way)
                        #     # img_feat = (img_feat[:, 0, :].float() + img_feat_flip[:, 0, :].float()) / 2

                        # elif input_img_type == InputImageType.color_pencil:
                        #     text_img_text_fu = model.fusion_layer(
                        #         text_feat, None, img_feat, None, imgs[-1], way=self.args.fusion_way)
                        #     # img_feat = (img_feat[:, 0, :].float() + img_feat_flip[:, 0, :].float()) / 2

                        # elif input_img_type == InputImageType.nir:
                        #     text_img_text_fu = model.fusion_layer(
                        #         text_feat, None, None, img_feat, imgs[-1], way=self.args.fusion_way)
                        # embeds.append(text_feat.cpu())

                        # text_feat_t = model.encode_text_tse(imgs[-1])
                        # embeds_t.append(text_feat_t.cpu())

                        # img_feat = img_feat[:, 0, :].float()
                        # if getattr(self.args, 'denoise_eval', False):
                        #     den_i = model.denoise_text_inference(img_feat, cond=None)
                        #     img_feat = self._blend_eval(img_feat, den_i)
                        embeds.append(img_feat.cpu())
                        # img_feat_t = model.encode_image_tse(imgs[0], input_img_type)
                        # embeds_1_t.append(img_feat_t.cpu())
                        # embeds_2.append(fusion_feat.cpu())

                        embeds_1.append(text_feat_origin.cpu())

                    else:
                        # 3种情况
                        if modalities[0] == 'sk' and modalities[1] == 'cp':
                            img_feat, _ = model.encode_image(imgs[0], InputImageType.sketch)
                            img2_feat, _ = model.encode_image(imgs[1], InputImageType.color_pencil)
                            # prefix_txt_token = tokenize(model.sk_caption_prefix, tokenizer=self.tokenizer, text_length=self.args.text_length, truncate=True).to(imgs[0].device)
                            # prefix_txt2_token = tokenize(model.cp_caption_prefix, tokenizer=self.tokenizer, text_length=self.args.text_length, truncate=True).to(imgs[0].device)

                            # _, _, image_features_proj, txt_global = model._use_inverse(prefix_txt_token.repeat(imgs[0].shape[0], 1),  imgs[0], InputImageType.sketch, use_prompt=True)
                            # img_feat, text_feat = image_features_proj[:, 0, :].float(), image_features_proj[:, -1, :].float()

                            # _, _, image_features_proj_2, txt_global_2 = model._use_inverse(prefix_txt2_token.repeat(imgs[1].shape[0], 1),  imgs[1], InputImageType.color_pencil, use_prompt=True)
                            # img2_feat, text_feat_2 = image_features_proj_2[:, 0, :].float(), image_features_proj_2[:, -1, :].float()
                            # img_feat_t = model.encode_image_tse(imgs[0], InputImageType.sketch)
                            # img2_feat_t = model.encode_image_tse(imgs[1], InputImageType.color_pencil)
                            # text, sk, cp, nir
                            # img_img_fu = model.fusion_layer(
                            #     None, img_feat, img2_feat, None, None, way=self.args.fusion_way)
                            # text_feat_1 = text_feat_sk
                            # text_feat_2 = text_feat_cp
                        elif modalities[0] == 'sk' and modalities[1] == 'nir':
                            img_feat, _ = model.encode_image(imgs[0], InputImageType.sketch)
                            img2_feat, _ = model.encode_image(imgs[1], InputImageType.nir)
                            # prefix_txt_token = tokenize(model.sk_caption_prefix, tokenizer=self.tokenizer, text_length=self.args.text_length, truncate=True).to(imgs[0].device)
                            # prefix_txt2_token = tokenize(model.nir_caption_prefix, tokenizer=self.tokenizer, text_length=self.args.text_length, truncate=True).to(imgs[0].device)

                            # _, _, image_features_proj, txt_global = model._use_inverse(prefix_txt_token.repeat(imgs[0].shape[0], 1),  imgs[0], InputImageType.sketch, use_prompt=True)
                            # img_feat, text_feat = image_features_proj[:, 0, :].float(), image_features_proj[:, -1, :].float()

                            # _, _, image_features_proj_2, txt_global_2 = model._use_inverse(prefix_txt2_token.repeat(imgs[1].shape[0], 1),  imgs[1], InputImageType.nir, use_prompt=True)
                            # img2_feat, text_feat_2 = image_features_proj_2[:, 0, :].float(), image_features_proj_2[:, -1, :].float()
                            # img_feat_t = model.encode_image_tse(imgs[0], InputImageType.sketch)
                            # img2_feat_t = model.encode_image_tse(imgs[1], InputImageType.nir)
                            # text, sk, cp, nir
                            # img_img_fu = model.fusion_layer(
                            #     None, img_feat, None, img2_feat, None, way=self.args.fusion_way)
                            # text_feat_1 = text_feat_sk
                            # text_feat_2 = text_feat_nir
                        elif modalities[0] == 'cp' and modalities[1] == 'nir':
                            img_feat, _ = model.encode_image(imgs[0], InputImageType.color_pencil)
                            img2_feat, _ = model.encode_image(imgs[1], InputImageType.nir)
                            # prefix_txt_token = tokenize(model.cp_caption_prefix, tokenizer=self.tokenizer, text_length=self.args.text_length, truncate=True).to(imgs[0].device)
                            # prefix_txt2_token = tokenize(model.nir_caption_prefix, tokenizer=self.tokenizer, text_length=self.args.text_length, truncate=True).to(imgs[0].device)

                            # _, _, image_features_proj, txt_global = model._use_inverse(prefix_txt_token.repeat(imgs[0].shape[0], 1),  imgs[0], InputImageType.color_pencil, use_prompt=True)
                            # img_feat, text_feat = image_features_proj[:, 0, :].float(), image_features_proj[:, -1, :].float()

                            # _, _, image_features_proj_2, txt_global_2 = model._use_inverse(prefix_txt2_token.repeat(imgs[1].shape[0], 1),  imgs[1], InputImageType.nir, use_prompt=True)
                            # img2_feat, text_feat_2 = image_features_proj_2[:, 0, :].float(), image_features_proj_2[:, -1, :].float()
                            # img_feat_t = model.encode_image_tse(imgs[0], InputImageType.color_pencil)
                            # img2_feat_t = model.encode_image_tse(imgs[1], InputImageType.nir)
                            # text, sk, cp, nir
                            # img_img_fu = model.fusion_layer(
                            #     None, None, img_feat, img2_feat, None, way=self.args.fusion_way)
                            # text_feat_1 = text_feat_cp
                            # text_feat_2 = text_feat_nir
                        # img_feat = img_feat[:, 0, :].float()
                        # img2_feat = img2_feat[:, 0, :].float()

                        # if getattr(self.args, 'denoise_eval', False):
                        #     den_1 = model.denoise_text_inference(img_feat, cond=None)
                        #     den_2 = model.denoise_text_inference(img2_feat, cond=None)
                        #     img_feat = self._blend_eval(img_feat, den_1)
                        #     img2_feat = self._blend_eval(img2_feat, den_2)
                        # fusion_feat = model._transformer_img_shared(img_feat + img2_feat)
                        embeds.append(img_feat.cpu())
                        embeds_1.append(img2_feat.cpu())
                        # embeds_t.append(img_feat_t.cpu())
                        # embeds_1_t.append(img2_feat_t.cpu())
                        # embeds_2.append(img_img_fu.cpu())
                        # embeds_2.append(fusion_feat.cpu())
                        # embeds_3.append(text_feat_2.cpu())

                        # embeds_4.append(txt_global.cpu())
                        # embeds_5.append(txt_global_2.cpu())

                elif len(modalities) == 3:
                    if 'text' in modalities:
                        # caption在base.py中统一放到了最后面
                        # text_feat = model.encode_text(imgs[-1])
                        # text_feat = self._get_new_text_feat_with_only_text(imgs[-1], model)

                        text_feat_origin = model.encode_text(imgs[-1], use_prompt=False)
                        text_feat_origin = text_feat_origin[torch.arange(text_feat_origin.shape[0]), imgs[-1].argmax(dim=-1)].float()

                        # text_feat_t = model.encode_text_tse(imgs[-1])
                        if modalities[1] == 'cp' and modalities[2] == 'sk':
                            img_feat, _ = model.encode_image(imgs[0], InputImageType.color_pencil)
                            img2_feat, _ = model.encode_image(imgs[1], InputImageType.sketch)
                            # img_feat_t = model.encode_image_tse(imgs[0], InputImageType.color_pencil)
                            # img2_feat_t = model.encode_image_tse(imgs[1], InputImageType.sketch)
                            # text, sk, cp, nir
                            # img_img_fu = model.fusion_layer(
                            #     text_feat, img2_feat, img_feat, None, imgs[-1], way=self.args.fusion_way)
                            # _, _, image_features_proj, txt_global = model._use_inverse(
                            #     torch.stack(self._get_new_token(imgs[-1].cpu().tolist(), model.cp_caption_prefix), dim=0).to(imgs[0].device),
                            #     imgs[0], InputImageType.color_pencil, use_prompt=False) ## 
                            # img_feat, text_feat = image_features_proj[:, 0, :].float(), image_features_proj[:, -1, :].float()
                            # # text_feat = text_feat + cp_txt_global

                            # _, _, image_features_proj_2, txt_global_2 = model._use_inverse(
                            #     torch.stack(self._get_new_token(imgs[-1].cpu().tolist(), model.sk_caption_prefix), dim=0).to(imgs[0].device),
                            #     imgs[1], InputImageType.sketch, use_prompt=False)
                            # img2_feat, text_feat_2 = image_features_proj_2[:, 0, :].float(), image_features_proj_2[:, -1, :].float()
                            # text_feat_2 = text_feat_2 + sk_txt_global

                        elif modalities[1] == 'cp' and modalities[2] == 'nir':
                            img_feat, _ = model.encode_image(imgs[0], InputImageType.color_pencil)
                            img2_feat, _ = model.encode_image(imgs[1], InputImageType.nir)
                            # img_feat_t = model.encode_image_tse(imgs[0], InputImageType.color_pencil)
                            # img2_feat_t = model.encode_image_tse(imgs[1], InputImageType.nir)
                            # text, sk, cp, nir
                            # img_img_fu = model.fusion_layer(
                            #     text_feat, None, img_feat, img2_feat, imgs[-1], way=self.args.fusion_way)

                            # _, _, image_features_proj, txt_global = model._use_inverse(
                            #     torch.stack(self._get_new_token(imgs[-1].cpu().tolist(), model.cp_caption_prefix), dim=0).to(imgs[0].device),
                            #     imgs[0], InputImageType.color_pencil, use_prompt=False)
                            # img_feat, text_feat = image_features_proj[:, 0, :].float(), image_features_proj[:, -1, :].float()
                            # # text_feat = text_feat + cp_txt_global

                            # _, _, image_features_proj_2, txt_global_2 = model._use_inverse(
                            #     torch.stack(self._get_new_token(imgs[-1].cpu().tolist(), model.nir_caption_prefix), dim=0).to(imgs[0].device),
                            #     imgs[1], InputImageType.nir, use_prompt=False)
                            # img2_feat, text_feat_2 = image_features_proj_2[:, 0, :].float(), image_features_proj_2[:, -1, :].float()
                            # text_feat_2 = text_feat_2 + nir_txt_global

                        elif modalities[1] == 'sk' and modalities[2] == 'nir':
                            img_feat, _ = model.encode_image(imgs[0], InputImageType.sketch)
                            img2_feat, _ = model.encode_image(imgs[1], InputImageType.nir)
                            # img_feat_t = model.encode_image_tse(imgs[0], InputImageType.sketch)
                            # img2_feat_t = model.encode_image_tse(imgs[1], InputImageType.nir)
                            # text, sk, cp, nir
                            # img_img_fu = model.fusion_layer(
                            #     text_feat, img_feat, None, img2_feat, imgs[-1], way=self.args.fusion_way)

                            # _, _, image_features_proj, txt_global = model._use_inverse(
                            #     torch.stack(self._get_new_token(imgs[-1].cpu().tolist(), model.sk_caption_prefix), dim=0).to(imgs[0].device),
                            #     imgs[0], InputImageType.sketch, use_prompt=False)
                            # img_feat, text_feat = image_features_proj[:, 0, :].float(), image_features_proj[:, -1, :].float()
                            # # text_feat = text_feat + sk_txt_global

                            # _, _, image_features_proj_2, txt_global_2 = model._use_inverse(
                            #     torch.stack(self._get_new_token(imgs[-1].cpu().tolist(), model.nir_caption_prefix), dim=0).to(imgs[0].device),
                            #     imgs[1], InputImageType.nir, use_prompt=False)
                            # img2_feat, text_feat_2 = image_features_proj_2[:, 0, :].float(), image_features_proj_2[:, -1, :].float()
                            # text_feat_2 = text_feat_2 + nir_txt_global
                        # img_feat = img_feat[:, 0, :].float()
                        # img2_feat = img2_feat[:, 0, :].float()
                        
                        # text_feat = text_feat[torch.arange(text_feat.shape[0]), imgs[-1].argmax(dim=-1)].float()

                        # embeds.append(text_feat.cpu())
                        embeds.append(text_feat_origin.cpu())
                        embeds_1.append(img_feat.cpu())
                        embeds_2.append(img2_feat.cpu())

                        # fusion_feat = model._transformer_img_shared(img_feat + img2_feat)

                        # embeds_t.append(text_feat_t.cpu())
                        # embeds_1_t.append(img_feat_t.cpu())
                        # embeds_2_t.append(img2_feat_t.cpu())
                        # embeds_3.append(fusion_feat.cpu())

                        # embeds_4.append(txt_global.cpu())
                        # embeds_5.append(txt_global_2.cpu())
                    else:
                        if modalities[0] == 'cp' and modalities[1] == 'sk' and modalities[2] == 'nir':
                            img_feat, _ = model.encode_image(imgs[0], InputImageType.color_pencil)
                            img2_feat, _ = model.encode_image(imgs[1], InputImageType.sketch)
                            img3_feat, _ = model.encode_image(imgs[2], InputImageType.nir)

                            # text_feat_1 = text_feat_cp
                            # text_feat_2 = text_feat_sk
                            # text_feat_3 = text_feat_nir

                            # img_feat_t = model.encode_image_tse(imgs[0], InputImageType.color_pencil)
                            # img2_feat_t = model.encode_image_tse(imgs[1], InputImageType.sketch)
                            # img3_feat_t = model.encode_image_tse(imgs[2], InputImageType.nir)
                            # text, sk, cp, nir
                            # img_img_fu = model.fusion_layer(
                            #     None, img2_feat, img_feat, img3_feat, None, way=self.args.fusion_way)

                            # img_feat = img_feat[:, 0, :].float()
                            # img2_feat = img2_feat[:, 0, :].float()
                            # img3_feat = img3_feat[:, 0, :].float()
                            embeds.append(img_feat.cpu())
                            embeds_1.append(img2_feat.cpu())
                            embeds_2.append(img3_feat.cpu())

                            # fusion_feat = model._transformer_img_shared(img_feat + img2_feat + img3_feat)

                            # embeds_3.append(fusion_feat.cpu())
                            # embeds_4.append(text_feat_2.cpu())
                            # embeds_5.append(text_feat_3.cpu())

                            # embeds_t.append(img_feat_t.cpu())
                            # embeds_1_t.append(img2_feat_t.cpu())
                            # embeds_2_t.append(img3_feat_t.cpu())
                            # embeds_3.append(img_img_fu.cpu())
                elif len(modalities) == 4:
                    # text_feat = model.encode_text(imgs[-1])
                    # text_feat = self._get_new_text_feat_with_only_text(imgs[-1], model)

                    text_feat_origin = model.encode_text(imgs[-1], use_prompt=False)
                    text_feat_origin = text_feat_origin[torch.arange(text_feat_origin.shape[0]), imgs[-1].argmax(dim=-1)].float()

                    img_feat, _ = model.encode_image(imgs[0], InputImageType.color_pencil)
                    img2_feat, _ = model.encode_image(imgs[1], InputImageType.sketch)
                    img3_feat, _ = model.encode_image(imgs[2], InputImageType.nir)

                    

                    # text_feat_t = model.encode_text_tse(imgs[-1])
                    # img_feat_t = model.encode_image_tse(imgs[0], InputImageType.color_pencil)
                    # img2_feat_t = model.encode_image_tse(imgs[1], InputImageType.sketch)
                    # img3_feat_t = model.encode_image_tse(imgs[2], InputImageType.nir)
                    # text_img_img_img_fu = model.fusion_layer(
                    #     text_feat, img2_feat, img_feat, img3_feat, imgs[-1], way=self.args.fusion_way  ### 注意顺序, 数据是 cp, sk ,nir, 但是fusion layer是 txt, sk ,cp , nir
                    # )


                    # text_feat = text_feat[torch.arange(text_feat.shape[0]), imgs[-1].argmax(dim=-1)].float()

                    # _, _, image_features_proj, cp_txt_global = model._use_inverse(
                    #     torch.stack(self._get_new_token(imgs[-1].cpu().tolist(), model.cp_caption_prefix), dim=0).to(imgs[0].device),
                    #     imgs[0], InputImageType.color_pencil, use_prompt=False)
                    # img_feat, text_feat = image_features_proj[:, 0, :].float(), image_features_proj[:, -1, :].float()
                    # # text_feat = text_feat + cp_txt_global

                    # _, _, image_features_proj_2, sk_txt_global = model._use_inverse(
                    #     torch.stack(self._get_new_token(imgs[-1].cpu().tolist(), model.sk_caption_prefix), dim=0).to(imgs[0].device),
                    #     imgs[1], InputImageType.sketch, use_prompt=False)
                    # img2_feat, text_feat_2 = image_features_proj_2[:, 0, :].float(), image_features_proj_2[:, -1, :].float()
                    # # text_feat_2 = text_feat_2 + sk_txt_global

                    # _, _, image_features_proj_3, nir_txt_global = model._use_inverse(
                    #     torch.stack(self._get_new_token(imgs[-1].cpu().tolist(), model.nir_caption_prefix), dim=0).to(imgs[0].device),
                    #     imgs[2], InputImageType.nir, use_prompt=False)
                    # img3_feat, text_feat_3 = image_features_proj_3[:, 0, :].float(), image_features_proj_3[:, -1, :].float()
                    # text_feat_3 = text_feat_3 + nir_txt_global

                    # embeds.append(text_feat.cpu())
                    embeds.append(text_feat_origin.cpu())
                    embeds_1.append(img_feat.cpu())
                    embeds_2.append(img2_feat.cpu())
                    embeds_3.append(img3_feat.cpu())

                    # fusion_feat = model._transformer_img_shared(img_feat + img2_feat + img3_feat)


                    # embeds_t.append(text_feat_t.cpu())
                    # embeds_1_t.append(img_feat_t.cpu())
                    # embeds_2_t.append(img2_feat_t.cpu())
                    # embeds_3_t.append(img3_feat_t.cpu())
                    # embeds_4.append(fusion_feat.cpu())
                    # embeds_5.append(text_feat_3.cpu())
                    
                # for i, modality in enumerate(modalities):
                #     encoder_method = getattr(model, self.modality_embed_encoders[modality])

                #     if modality == 'text':
                #         # Text encoding returns tuple, we take the second element
                #         _, text_embed = encoder_method(imgs[i])
                #         embeds.append(text_embed)
                #     else:
                #         embed = encoder_method(imgs[i])
                #         embeds.append(embed)

                # # Concatenate all embeddings and fuse (reid5o只用融合特征计算相似度)
                # combined = torch.cat(embeds, dim=1)
                # fusion_feats = model.mm_fusion(combined, combined, combined)

            qids.append(pid.view(-1).cpu())
            # qfeats.append(fusion_feats)

        qids = torch.cat(qids, 0)
        # qfeats = torch.cat(qfeats, 0)
        # qfeats = F.normalize(qfeats, p=2, dim=1)
        qfeats = F.normalize(torch.cat(embeds, 0), p=2, dim=1)
        qfeats_1 = F.normalize(torch.cat(embeds_1, 0), p=2, dim=1) if len(embeds_1) > 0 else None
        qfeats_2 = F.normalize(torch.cat(embeds_2, 0), p=2, dim=1) if len(embeds_2) > 0 else None
        qfeats_3 = F.normalize(torch.cat(embeds_3, 0), p=2, dim=1) if len(embeds_3) > 0 else None
        qfeats_4 = F.normalize(torch.cat(embeds_4, 0), p=2, dim=1) if len(embeds_4) > 0 else None
        qfeats_5 = F.normalize(torch.cat(embeds_5, 0), p=2, dim=1) if len(embeds_5) > 0 else None

        # qfeats_t = F.normalize(torch.cat(embeds_t, 0), p=2, dim=1)
        # qfeats_1_t = F.normalize(torch.cat(embeds_1_t, 0), p=2, dim=1) if len(embeds_1_t) > 0 else None
        # qfeats_2_t = F.normalize(torch.cat(embeds_2_t, 0), p=2, dim=1) if len(embeds_2_t) > 0 else None
        # qfeats_3_t = F.normalize(torch.cat(embeds_3_t, 0), p=2, dim=1) if len(embeds_3_t) > 0 else None
        # qfeats_4_t = F.normalize(torch.cat(embeds_4_t, 0), p=2, dim=1) if len(embeds_4_t) > 0 else None

        return qids, qfeats, qfeats_1, qfeats_2, qfeats_3, qfeats_4, qfeats_5, qimage_ids#, qfeats_t, qfeats_1_t, qfeats_2_t, qfeats_3_t, qfeats_4_t

    def _evaluate_modality_rnt(self, gfeats, gids, gcamids,  model, loader, modalities,
                              gfeats_rgb=None, gfeats_nir=None, gfeats_tir=None, task_name=None):
        ###
        ### 2026.0601
        ### gallery 替换为 nir

        model = model.eval()
        device = next(model.parameters()).device
        qids, qfeats, qcamids = [], [], []
        qfeats_rgb, qfeats_nir, qfeats_tir = [], [], []
        # RGBNT201 常见的 “RGB-NI-TI” 多模态检索：
        # 1) concat：query/gallery 都使用三模态特征拼接
        # 2) per_modality_avg：分别算 RGB/NI/TI 三个检索指标再平均

        for pids, imgs, camids, texts in loader:  ## TODO text加入进来
            imgs = {k : v.to(device) for k,v in imgs.items()}
            # texts_rgb = texts['rgb_text'].to(device)
            # 2026.0601: gallery替换为Nir, text应该用nir
            texts_rgb = texts['rgb_text'].to(device)
            with torch.no_grad():
                # def forward_query():
                #     rgb_feat, _ = model.encode_image(imgs['RGB'], InputImageType.rgb)
                #     nir_feat, _ = model.encode_image(imgs['NI'], InputImageType.nir)
                #     tir_feat, _ = model.encode_image(imgs['TI'], InputImageType.color_pencil)
                #     text_feat = model.encode_text(texts_rgb, modality=InputImageType.rgb, use_prompt=False)
                #     text_feat = self._text_feature_from_tokens(text_feat, texts_rgb)
                #     return rgb_feat, nir_feat, tir_feat, text_feat

                # (rgb_feat, nir_feat, tir_feat, text_feat), elapsed_ms = measure_forward_latency(forward_query, device)

                rgb_feat, _ = model.encode_image(imgs['RGB'], InputImageType.rgb)
                nir_feat, _ = model.encode_image(imgs['NI'], InputImageType.nir)
                tir_feat, _ = model.encode_image(imgs['TI'], InputImageType.color_pencil)
                text_feat = model.encode_text(texts_rgb, modality=InputImageType.rgb, use_prompt=False)  # modality对于Text是没有用的
                text_feat = text_feat[torch.arange(text_feat.shape[0]), texts_rgb.argmax(dim=-1)].float()
                # if self.rnt_eval_mode == "concat":
                #     q_feat = torch.cat([rgb_feat, nir_feat, tir_feat], dim=1)
                #     qfeats.append(q_feat.cpu())
                # else:
                qfeats_rgb.append(rgb_feat.cpu())
                qfeats_nir.append(nir_feat.cpu())
                qfeats_tir.append(tir_feat.cpu())
                qfeats.append(text_feat.cpu())
            # latency_meter = getattr(self, "query_latency_meter", None)
            # if latency_meter is not None:
            #     batch_size = pids.numel() if hasattr(pids, "numel") else len(pids)
            #     latency_meter.update(task_name, len(modalities), elapsed_ms, batch_size)
            qids.append(torch.tensor(pids).view(-1).cpu())
            qcamids.append(torch.tensor(camids).view(-1).cpu())

        qids = torch.cat(qids, 0)
        qcamids = torch.cat(qcamids, 0)

        # if self.rnt_eval_mode == "concat":
        #     qfeats = F.normalize(torch.cat(qfeats, 0), p=2, dim=1)

        #     # 欧式距离（Market 风格）评测
        #     m, n = qfeats.shape[0], gfeats.shape[0]
        #     distmat = torch.pow(qfeats, 2).sum(dim=1, keepdim=True).expand(m, n) + \
        #               torch.pow(gfeats, 2).sum(dim=1, keepdim=True).expand(n, m).t()
        #     distmat.addmm_(1, -2, qfeats, gfeats.t())
        #     distmat = distmat.cpu().numpy()

        #     cmc, mAP = eval_func(distmat, qids.numpy(), gids.numpy(), qcamids.numpy(), gcamids.numpy())
        #     return cmc[0], cmc[4], cmc[9], mAP, 0   ## minp就不算了

        # if self.rnt_eval_mode != "per_modality_avg":
        #     raise ValueError(f"Unknown rnt_eval_mode: {self.rnt_eval_mode}. Expected 'concat' or 'per_modality_avg'.")

        # # per_modality_avg：分别算 RGB/NI/TI 三套指标后平均
        # if gfeats_rgb is None or gfeats_nir is None or gfeats_tir is None:
        #     raise ValueError("per_modality_avg requires gfeats_rgb/gfeats_nir/gfeats_tir (gallery feats for each modality).")

        q_text = F.normalize(torch.cat(qfeats, 0), p=2, dim=1)
        q_nir = F.normalize(torch.cat(qfeats_nir, 0), p=2, dim=1)
        q_tir = F.normalize(torch.cat(qfeats_tir, 0), p=2, dim=1)
        q_rgb = F.normalize(torch.cat(qfeats_rgb, 0), p=2, dim=1)

        def _euclidean(q, g):
            m, n = q.shape[0], g.shape[0]
            dist = torch.pow(q, 2).sum(dim=1, keepdim=True).expand(m, n) + \
                   torch.pow(g, 2).sum(dim=1, keepdim=True).expand(n, m).t()
            dist.addmm_(1, -2, q, g.t())
            return dist.cpu().numpy()

        if len(modalities) == 1:
            if 'ti' in modalities:
                cmc, mAP, mINP = eval_func(_euclidean(q_tir, gfeats_rgb), qids.numpy(), gids.numpy(), qcamids.numpy(), gcamids.numpy(), set=2, max_rank=10)
            elif 'nir' in modalities:
                cmc, mAP, mINP = eval_func(_euclidean(q_nir, gfeats_rgb), qids.numpy(), gids.numpy(), qcamids.numpy(), gcamids.numpy(), set=2, max_rank=10)
            elif 'text' in modalities:
                cmc, mAP, mINP = eval_func(_euclidean(q_text, gfeats_rgb), qids.numpy(), gids.numpy(), qcamids.numpy(), gcamids.numpy(), set=2, max_rank=10)
            elif 'rgb' in modalities:
                cmc, mAP, mINP = eval_func(_euclidean(q_rgb, gfeats_rgb), qids.numpy(), gids.numpy(), qcamids.numpy(), gcamids.numpy(), set=2, max_rank=10)
        elif len(modalities) == 2:
            if 'ti' in modalities and 'nir' in modalities:
                cmc, mAP, mINP = eval_func((_euclidean(q_tir, gfeats_rgb) + _euclidean(q_nir, gfeats_rgb)) / 2, qids.numpy(), gids.numpy(), qcamids.numpy(), gcamids.numpy(), set=2, max_rank=10)
            elif 'ti' in modalities and 'text' in modalities:
                cmc, mAP, mINP = eval_func((_euclidean(q_tir, gfeats_rgb) + _euclidean(q_text, gfeats_rgb)) / 2, qids.numpy(), gids.numpy(), qcamids.numpy(), gcamids.numpy(), set=2, max_rank=10)
            elif 'nir' in modalities and 'text' in modalities:
                cmc, mAP, mINP = eval_func((_euclidean(q_nir, gfeats_rgb) + _euclidean(q_text, gfeats_rgb)) / 2, qids.numpy(), gids.numpy(), qcamids.numpy(), gcamids.numpy(), set=2, max_rank=10)
            
            elif 'rgb' in modalities and 'nir' in modalities:
                cmc, mAP, mINP = eval_func((_euclidean(q_rgb, gfeats_rgb) + _euclidean(q_nir, gfeats_rgb)) / 2, qids.numpy(), gids.numpy(), qcamids.numpy(), gcamids.numpy(), set=2, max_rank=10)
            elif 'rgb' in modalities and 'ti' in modalities:
                cmc, mAP, mINP = eval_func((_euclidean(q_rgb, gfeats_rgb) + _euclidean(q_tir, gfeats_rgb)) / 2, qids.numpy(), gids.numpy(), qcamids.numpy(), gcamids.numpy(), set=2, max_rank=10)
            elif 'rgb' in modalities and 'text' in modalities:
                cmc, mAP, mINP = eval_func((_euclidean(q_rgb, gfeats_rgb) + _euclidean(q_text, gfeats_rgb)) / 2, qids.numpy(), gids.numpy(), qcamids.numpy(), gcamids.numpy(), set=2, max_rank=10)

        elif len(modalities) == 3:
            if 'ti' in modalities and 'nir' in modalities and 'text' in modalities:
                cmc, mAP, mINP = eval_func((_euclidean(q_tir, gfeats_rgb) + _euclidean(q_nir, gfeats_rgb) + _euclidean(q_text, gfeats_rgb)) / 3, 
                                        qids.numpy(), gids.numpy(), qcamids.numpy(), gcamids.numpy(), set=2, max_rank=10)
            elif 'ti' in modalities and 'nir' in modalities and 'rgb' in modalities:
                cmc, mAP, mINP = eval_func((_euclidean(q_tir, gfeats_rgb) + _euclidean(q_nir, gfeats_rgb) + _euclidean(q_rgb, gfeats_rgb)) / 3, 
                                        qids.numpy(), gids.numpy(), qcamids.numpy(), gcamids.numpy(), set=2, max_rank=10)
            elif 'rgb' in modalities and 'nir' in modalities and 'text' in modalities:
                cmc, mAP, mINP = eval_func((_euclidean(q_rgb, gfeats_rgb) + _euclidean(q_nir, gfeats_rgb) + _euclidean(q_text, gfeats_rgb)) / 3, 
                                        qids.numpy(), gids.numpy(), qcamids.numpy(), gcamids.numpy(), set=2, max_rank=10)
            elif 'rgb' in modalities and 'ti' in modalities and 'text' in modalities:
                cmc, mAP, mINP = eval_func((_euclidean(q_rgb, gfeats_rgb) + _euclidean(q_tir, gfeats_rgb) + _euclidean(q_text, gfeats_rgb)) / 3, 
                                        qids.numpy(), gids.numpy(), qcamids.numpy(), gcamids.numpy(), set=2, max_rank=10)
        elif len(modalities) == 4:
            cmc, mAP, mINP = eval_func((_euclidean(q_tir, gfeats_rgb) + _euclidean(q_nir, gfeats_rgb) + _euclidean(q_text, gfeats_rgb) + _euclidean(q_rgb, gfeats_rgb)) / 4, 
                                        qids.numpy(), gids.numpy(), qcamids.numpy(), gcamids.numpy(), set=2, max_rank=10)
        else:
            raise TypeError('The modalities is not equal to 1/2/3 in RNT, please check your datasets!')
        ## 三个模态单独作为query
        # cmc_rgb, mAP_rgb = eval_func(_euclidean(q_rgb, gfeats_rgb), qids.numpy(), gids.numpy(), qcamids.numpy(), gcamids.numpy())
        
        
        return cmc[0], cmc[4], cmc[9], mAP, mINP 
        
        # r1 = (cmc_rgb[0] + cmc_nir[0] + cmc_tir[0]) / 3.0
        # r5 = (cmc_rgb[4] + cmc_nir[4] + cmc_tir[4]) / 3.0
        # r10 = (cmc_rgb[9] + cmc_nir[9] + cmc_tir[9]) / 3.0
        # mAP = (mAP_rgb + mAP_nir + mAP_tir) / 3.0
        # return r1, r5, r10, mAP, 0


    def _evaluate_modality(self, gfeats, gids, model, loader, modalities, gimage_ids=None, task_name=None):
        """Generic evaluation function for any modality combination"""
        # if len(modalities) == 1:
        #     qids, qfeats = self._extract_single_modality_features(model, loader, modalities[0])
        # else:
        qids, qfeats, qfeats_1, qfeats_2, qfeats_3, qfeats_4, qfeats_5, qimage_ids = self._extract_multi_modality_features_with_forward_latency(model, loader, modalities, task_name)

        similarity = qfeats @ gfeats.t()
        # similarity_t = qfeats_t @ gfeats_t.t()
        # similarity = (similarity + similarity_t) / 2
        q_num = 1
        if qfeats_1 is not None:
            # if len(modalities) == 1 and modalities[0] != 'text':
            #     sim_1 = qfeats_1 @ gfeats_t.t()
            # else:
            sim_1 = qfeats_1 @ gfeats.t()
            # sim_1_t = qfeats_1_t @ gfeats_t.t()
            # similarity += (sim_1 + sim_1_t) / 2
            similarity += sim_1
            q_num += 1

        if qfeats_2 is not None:
            sim_2 = qfeats_2 @ gfeats.t()
            # sim_2_t = qfeats_2_t @ gfeats_t.t()
            # similarity += (sim_2 + sim_2_t) / 2
            similarity += sim_2
            q_num += 1

        if qfeats_3 is not None:
            sim_3 = qfeats_3 @ gfeats.t()
            # sim_3_t = qfeats_3_t @ gfeats_t.t()
            # similarity += (sim_3 + sim_3_t) / 2
            similarity += sim_3
            q_num += 1

        if qfeats_4 is not None:
            sim_4 = qfeats_4 @ gfeats.t()
            # sim_4_t = qfeats_4_t @ gfeats_t.t()
            # similarity += (sim_4 + sim_4_t) / 2
            similarity += sim_4
            q_num += 1
        
        if qfeats_5 is not None:
            sim_5 = qfeats_5 @ gfeats.t()
            # sim_4_t = qfeats_4_t @ gfeats_t.t()
            # similarity += (sim_4 + sim_4_t) / 2
            similarity += sim_5
            q_num += 1
        similarity = similarity / q_num


        if gimage_ids is not None and len(qimage_ids) > 0:
            # # remove the rgb images that used for generated sketches from gallery set
            if self.args.dataset_name == 'PKU-Sketch':
                input_set = 0
            else:
                input_set = 2
            # 与unireid的评测基准一致
            qimage_ids = torch.cat(qimage_ids, 0)
            t2i_cmc, t2i_mAP, t2i_mINP = eval_func(-similarity.numpy() , qids.numpy(), gids.numpy(), qimage_ids.numpy(), gimage_ids.numpy(), 
                                                    set=input_set, max_rank=10)  ##set =2 去除相同image id的样本评测 => 只有tri cuhk, tri icfg和tri rstp; pku正常设置set=0
        else:
            t2i_cmc, t2i_mAP, t2i_mINP, _ = rank(
                similarity=similarity,
                q_pids=qids,
                g_pids=gids,
                max_rank=10,
                get_mAP=self.get_mAP
            )
            t2i_cmc, t2i_mAP, t2i_mINP = t2i_cmc.cpu().numpy(), t2i_mAP.cpu().numpy(), t2i_mINP.cpu().numpy()

        return t2i_cmc[0], t2i_cmc[4], t2i_cmc[9], t2i_mAP, t2i_mINP

    def _get_modality_combinations(self):
        """Define all modality combinations and their corresponding loaders"""

        # 合并相同的, 顺序全部统一, text名称在最前面
        # orbench.py已经合并了相同的datasets, 顺序不同调换即可
        return [
            # Single modalities
            ('NIR', ['nir'], self.query_loaders.get('nir_query_loader')),
            ('CP', ['cp'], self.query_loaders.get('cp_query_loader')),
            ('SK', ['sk'], self.query_loaders.get('sk_query_loader')),
            ('TEXT', ['text'], self.query_loaders.get('text_query_loader')),

            # Two modalities
            # ('NIR+CP', ['nir', 'cp'], self.query_loaders.get('nir_cp_query_loader')),
            ('CP+NIR', ['cp', 'nir'], self.query_loaders.get('cp_nir_query_loader')),
            # ('NIR+SK', ['nir', 'sk'], self.query_loaders.get('nir_sk_query_loader')),
            ('SK+NIR', ['sk', 'nir'], self.query_loaders.get('sk_nir_query_loader')),
            # ('NIR+TEXT', ['nir', 'text'], self.query_loaders.get('nir_text_query_loader')),
            ('TEXT+NIR', ['text', 'nir'], self.query_loaders.get('text_nir_query_loader')),
            # ('CP+SK', ['cp', 'sk'], self.query_loaders.get('cp_sk_query_loader')),
            ('SK+CP', ['sk', 'cp'], self.query_loaders.get('sk_cp_query_loader')),
            # ('CP+TEXT', ['cp', 'text'], self.query_loaders.get('cp_text_query_loader')),
            ('TEXT+CP', ['text', 'cp'], self.query_loaders.get('text_cp_query_loader')),
            # ('SK+TEXT', ['sk', 'text'], self.query_loaders.get('sk_text_query_loader')),
            ('TEXT+SK', ['text', 'sk'], self.query_loaders.get('text_sk_query_loader')),

            # Three modalities
            # ('NIR+CP+SK', ['nir', 'cp', 'sk'], self.query_loaders.get('nir_cp_sk_query_loader')),
            # ('CP+NIR+SK', ['cp', 'nir', 'sk'], self.query_loaders.get('cp_nir_sk_query_loader')),
            # ('SK+NIR+CP', ['sk', 'nir', 'cp'], self.query_loaders.get('sk_nir_cp_query_loader')),
            ('CP+SK+NIR', ['cp', 'sk', 'nir'], self.query_loaders.get('cp_sk_nir_query_loader')),
            # ('NIR+CP+TEXT', ['nir', 'cp', 'text'], self.query_loaders.get('nir_cp_text_query_loader')),
            # ('CP+NIR+TEXT', ['cp', 'nir', 'text'], self.query_loaders.get('cp_nir_text_query_loader')),
            ('TEXT+CP+NIR', ['text', 'cp', 'nir'], self.query_loaders.get('text_cp_nir_query_loader')),
            # ('NIR+SK+TEXT', ['nir', 'sk', 'text'], self.query_loaders.get('nir_sk_text_query_loader')),
            # ('SK+NIR+TEXT', ['sk', 'nir', 'text'], self.query_loaders.get('sk_nir_text_query_loader')),
            ('TEXT+SK+NIR', ['text', 'sk', 'nir'], self.query_loaders.get('text_sk_nir_query_loader')),
            # ('CP+SK+TEXT', ['cp', 'sk', 'text'], self.query_loaders.get('cp_sk_text_query_loader')),
            # ('SK+CP+TEXT', ['sk', 'cp', 'text'], self.query_loaders.get('sk_cp_text_query_loader')),
            ('TEXT+CP+SK', ['text', 'cp', 'sk'], self.query_loaders.get('text_cp_sk_query_loader')),

            # Four modalities
            # ('NIR+CP+SK+TEXT', ['nir', 'cp', 'sk', 'text'], self.query_loaders.get('nir_cp_sk_text_query_loader')),
            # ('CP+NIR+SK+TEXT', ['cp', 'nir', 'sk', 'text'], self.query_loaders.get('cp_nir_sk_text_query_loader')),
            # ('SK+NIR+CP+TEXT', ['sk', 'nir', 'cp', 'text'], self.query_loaders.get('sk_nir_cp_text_query_loader')),
            # ('TEXT+NIR+CP+SK', ['text', 'nir', 'cp', 'sk'], self.query_loaders.get('text_nir_cp_sk_query_loader')),
            ('TEXT+CP+SK+NIR', ['text', 'cp', 'sk', 'nir'], self.query_loaders.get('text_cp_sk_nir_query_loader')),
        ]

    def _get_modality_combinations_tri(self):
        """tri cuhk, tri rstp, tri icfg"""
        return [
            # Single modalities
            ('SK', ['sk'], self.query_loaders.get('sk_query_loader')),
            ('TEXT', ['text'], self.query_loaders.get('text_query_loader')),
            # Two modalities
            ('TEXT+SK', ['text', 'sk'], self.query_loaders.get('text_sk_query_loader')),
        ]

    def _get_modality_combinations_rnt(self):
        """RNT"""
        return [
            # Single modalities
            ('RGB', ['rgb'], self.query_loaders.get('nir_query_loader')),  ### nir_query_loader作为占位符, 实际上是包含了三个模态
            ('NI', ['nir'], self.query_loaders.get('nir_query_loader')),
            ('TI', ['ti'], self.query_loaders.get('nir_query_loader')),
            ('TEXT', ['text'], self.query_loaders.get('nir_query_loader')),

            # two modalities
            ('NI+TI', ['nir', 'ti'], self.query_loaders.get('nir_query_loader')),
            ('NI+TEXT', ['nir', 'text'], self.query_loaders.get('nir_query_loader')),
            ('TI+TEXT', ['ti', 'text'], self.query_loaders.get('nir_query_loader')),

            ('RGB+NI', ['rgb', 'nir'], self.query_loaders.get('nir_query_loader')),
            ('RGB+TI', ['rgb', 'ti'], self.query_loaders.get('nir_query_loader')),
            ('RGB+TEXT', ['rgb', 'text'], self.query_loaders.get('nir_query_loader')),

            # three modalities
            # ('NI+TI+TEXT', ['nir', 'ti', 'text'], self.query_loaders.get('nir_query_loader')),
            # ('RGB+TI+TEXT', ['rgb', 'ti', 'text'], self.query_loaders.get('nir_query_loader')),
            # ('RGB+NI+TEXT', ['rgb', 'nir', 'text'], self.query_loaders.get('nir_query_loader')),
            # ('NI+TI+RGB', ['nir', 'ti', 'rgb'], self.query_loaders.get('nir_query_loader')),

            # # four
            # ('NI+TI+RGB+TEXT', ['nir', 'ti', 'rgb', 'text'], self.query_loaders.get('nir_query_loader')),

        ]

    def eval(self, model):
        model = model.eval()
        device = next(model.parameters()).device

        # Extract gallery features
        gids, gfeats = [], []
        gfeats_t = []
        camids_all = []
        gimage_ids = []
        if self.args.dataset_name in ['RGBNT201', 'RGBNT201_Text']:
            # 额外缓存每个模态单独的 gallery feats，支持 per_modality_avg
            gids, camids_all = [], []
            gfeats_rgb_list, gfeats_nir_list, gfeats_tir_list = [], [], []
            for pids, imgs, camids, _ in self.gallery_loader:
                # 1.三种模态数据分别经过model
                imgs = {k : v.to(device) for k,v in imgs.items()}
                with torch.no_grad():
                    rgb_feat, _ = model.encode_image(imgs['RGB'], InputImageType.rgb)
                    nir_feat, _ = model.encode_image(imgs['NI'], InputImageType.nir)
                    tir_feat, _ = model.encode_image(imgs['TI'], InputImageType.color_pencil)

                    # 把三种模态特征拼接在一起作为gallery,符合RNT reid任务常用操作
                    img_feat = torch.cat([rgb_feat, nir_feat, tir_feat], dim=1)

                gids.append(torch.tensor(pids).view(-1).cpu())
                gfeats.append(img_feat.cpu())
                camids_all.append(torch.tensor(camids).view(-1).cpu())
                gfeats_rgb_list.append(rgb_feat.cpu())
                gfeats_nir_list.append(nir_feat.cpu())
                gfeats_tir_list.append(tir_feat.cpu())
            gids = torch.cat(gids, 0)
            gfeats = torch.cat(gfeats, 0)
            # gfeats_t = torch.cat(gfeats_t, 0)
            camids_all = torch.cat(camids_all, 0)
            gfeats = F.normalize(gfeats, p=2, dim=1)
            gfeats_rgb = F.normalize(torch.cat(gfeats_rgb_list, 0), p=2, dim=1)
            gfeats_nir = F.normalize(torch.cat(gfeats_nir_list, 0), p=2, dim=1)
            gfeats_tir = F.normalize(torch.cat(gfeats_tir_list, 0), p=2, dim=1)
        elif self.args.dataset_name in ['CUHK-PEDES', 'ICFG-PEDES', 'RSTPReid', 'PKU-Sketch']:
            for pid, img, image_id in self.gallery_loader:
                img = img.to(device)
                with torch.no_grad():
                    # img_feat = model.encode_rgb_cls(img)
                    img_feat, _ = model.encode_image(img, InputImageType.rgb)

                    # _,_, img_features = model.base_model.encode_image(img, InputImageType.rgb)
                    # img_feat = img_features[:, 0, :].float()

                gids.append(pid.view(-1).cpu())
                gfeats.append(img_feat.cpu())
                gimage_ids.append(image_id)

            gids = torch.cat(gids, 0)
            gfeats = torch.cat(gfeats, 0)
            gfeats = F.normalize(gfeats, p=2, dim=1)
            gimage_ids = torch.cat(gimage_ids, 0)
        else:
            for pid, img in self.gallery_loader:
                img = img.to(device)
                with torch.no_grad():
                    # img_feat = model.encode_rgb_cls(img)
                    img_feat, _ = model.encode_image(img, InputImageType.rgb)

                    # _,_, img_features = model.base_model.encode_image(img, InputImageType.rgb)
                    # img_feat = img_features[:, 0, :].float()

                gids.append(pid.view(-1).cpu())
                gfeats.append(img_feat.cpu())
                # gfeats_t.append(img_feat_t.cpu())
            gids = torch.cat(gids, 0)
            gfeats = torch.cat(gfeats, 0)
            # gfeats_t = torch.cat(gfeats_t, 0)
            gfeats = F.normalize(gfeats, p=2, dim=1)
            # gfeats_t = F.normalize(gfeats_t, p=2, dim=1)
            gimage_ids = None

        eval_results = {}
        if self.args.dataset_name == 'ORBench':
            modality_combinations = self._get_modality_combinations()
        elif self.args.dataset_name in ['CUHK-PEDES', 'ICFG-PEDES', 'RSTPReid', 'PKU-Sketch']:
            modality_combinations = self._get_modality_combinations_tri()
        elif self.args.dataset_name in ['RGBNT201', 'RGBNT201_Text']:
            modality_combinations = self._get_modality_combinations_rnt()

        # Group evaluations by modality count for organized printing
        modality_groups = {
            1: "One Modality Evaluating...",
            2: "Two Modalities Evaluating...",
            3: "Three Modalities Evaluating...",
            4: "Four Modalities Evaluating..."
        }

        current_modality_count = 0
        self.query_latency_meter = QueryLatencyMeter()

        # 遍历每一种模态组合
        for task_name, modalities, loader in modality_combinations:
            if loader is None:
                continue

            modality_count = len(modalities)
            if modality_count != current_modality_count:
                current_modality_count = modality_count
                self.logger.info(modality_groups.get(modality_count, ""))

            if self.args.dataset_name in ['RGBNT201', 'RGBNT201_Text']:
                result = self._evaluate_modality_rnt(
                    gfeats, gids, camids_all, model, loader, modalities,
                    gfeats_rgb=gfeats_rgb if 'gfeats_rgb' in locals() else None,
                    gfeats_nir=gfeats_nir if 'gfeats_nir' in locals() else None,
                    gfeats_tir=gfeats_tir if 'gfeats_tir' in locals() else None,
                    task_name=task_name,
                )
            else:
                result = self._evaluate_modality(gfeats,  gids, model, loader, modalities, gimage_ids=gimage_ids, task_name=task_name)
            eval_results[task_name] = result

            self.logger.info(
                f"{task_name}: R1={result[0]:.3f}, R5={result[1]:.3f}, "
                f"R10={result[2]:.3f}, mAP={result[3]:.3f}, mINP={result[4]:.3f}"
            )

        # Build summary table
        table = PrettyTable(["task", "R1", "R5", "R10", "mAP", "mINP"])
        for task_name, _, _ in modality_combinations:
            if task_name in eval_results:
                result = eval_results[task_name]
                table.add_row([task_name, result[0], result[1], result[2], result[3], result[4]])

        # Calculate averages
        if self.args.dataset_name == 'ORBench':
            one_aver = self.table_average_calculation(table, 0, 4)
            two_aver = self.table_average_calculation(table, 4, 10)
            three_aver = self.table_average_calculation(table, 10, 14)
            four_aver = self.table_average_calculation(table, 14, 15)
            table.add_row(['ONE_AVER', one_aver[0], one_aver[1], one_aver[2], one_aver[3], one_aver[4]])
            table.add_row(['TWO_AVER', two_aver[0], two_aver[1], two_aver[2], two_aver[3], two_aver[4]])
            table.add_row(['THREE_AVER', three_aver[0], three_aver[1], three_aver[2], three_aver[3], three_aver[4]])
            table.add_row(['FOUR_AVER', four_aver[0], four_aver[1], four_aver[2], four_aver[3], four_aver[4]])

        elif self.args.dataset_name in ['CUHK-PEDES', 'ICFG-PEDES', 'RSTPReid', 'PKU-Sketch']:
            one_aver = self.table_average_calculation(table, 0, 2)
            two_aver = self.table_average_calculation(table, 2, 3)
            table.add_row(['ONE_AVER', one_aver[0], one_aver[1], one_aver[2], one_aver[3], one_aver[4]])
            table.add_row(['TWO_AVER', two_aver[0], two_aver[1], two_aver[2], two_aver[3], two_aver[4]])

        elif self.args.dataset_name in ['RGBNT201', 'RGBNT201_Text']:
            one_aver = self.table_average_calculation(table, 0, 4)
            two_aver = self.table_average_calculation(table, 4, 10)
            # three_aver = self.table_average_calculation(table, 10, 14)
            # four_aver = self.table_average_calculation(table, 14, 15)
            table.add_row(['ONE_AVER', one_aver[0], one_aver[1], one_aver[2], one_aver[3], one_aver[4]])
            table.add_row(['TWO_AVER', two_aver[0], two_aver[1], two_aver[2], two_aver[3], two_aver[4]])
            # table.add_row(['THREE_AVER', three_aver[0], three_aver[1], three_aver[2], three_aver[3], three_aver[4]])
            # table.add_row(['FOUR_AVER', four_aver[0], four_aver[1], four_aver[2], four_aver[3], four_aver[4]])

        # Format table
        for field in ["R1", "R5", "R10", "mAP", "mINP"]:
            table.custom_format[field] = lambda f, v: f"{v:.3f}"

        self.logger.info('\n' + str(table))
        self.query_latency_meter.log(self.logger)
        self.query_latency_meter = None

        if self.args.dataset_name == 'ORBench':
            return (one_aver[3] + two_aver[3] + three_aver[3] + four_aver[3]) / 4  ### 返回的是 map
        elif self.args.dataset_name in ['CUHK-PEDES', 'ICFG-PEDES', 'RSTPReid', 'PKU-Sketch']:
            return (one_aver[3] + two_aver[3]) / 2
        elif self.args.dataset_name in ['RGBNT201', 'RGBNT201_Text']:
            # return (one_aver[3] + two_aver[3] + three_aver[3] + four_aver[3]) / 4  ### 返回的是 map,  nir, tir, text, rgb(RNT数据集把rgb作为query之一了)
            return (one_aver[3] + two_aver[3]) / 2  ### 仅计算单模态和双模态
        

    def table_average_calculation(self, table, first, last):
        selected_rows = table._rows[first:last]
        column_sums = [0] * (len(table.field_names) - 1)
        num_rows = len(selected_rows)

        for row in selected_rows:
            for i, value in enumerate(row[1:]):
                column_sums[i] += value

        averages = [sum_val / num_rows for sum_val in column_sums]
        return averages
