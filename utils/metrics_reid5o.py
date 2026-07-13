from prettytable import PrettyTable
import torch
import torch.nn.functional as F
import logging
from typing import List, Tuple, Callable, Dict, Any

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


class Evaluator():
    def __init__(self, args, gallery_loader, get_mAP=True, **query_loaders):
        self.args = args
        self.gallery_loader = gallery_loader
        self.query_loaders = query_loaders
        self.get_mAP = get_mAP
        self.logger = logging.getLogger("ORBench.eval")

        # Define modality encoding methods
        self.modality_encoders = {
            'nir': 'encode_nir_cls',
            'cp': 'encode_cp_cls',
            'sk': 'encode_sk_cls',
            'text': 'encode_text_cls'
        }

        self.modality_embed_encoders = {
            'nir': 'encode_nir_embeds',
            'cp': 'encode_cp_embeds',
            'sk': 'encode_sk_embeds',
            'text': 'encode_text_embeds'
        }

    def _extract_single_modality_features(self, model, loader, modality):
        """Extract features for single modality"""
        model = model.eval()
        device = next(model.parameters()).device
        qids, qfeats = [], []

        for pid, img in loader:
            img = img.to(device)
            with torch.no_grad():
                encoder_method = getattr(model, self.modality_encoders[modality])
                img_feat = encoder_method(img)
            qids.append(pid.view(-1))
            qfeats.append(img_feat)

        qids = torch.cat(qids, 0)
        qfeats = torch.cat(qfeats, 0)
        qfeats = F.normalize(qfeats, p=2, dim=1)

        return qids, qfeats

    def _extract_multi_modality_features(self, model, loader, modalities):
        """Extract fused features for multiple modalities"""
        model = model.eval()
        device = next(model.parameters()).device
        qids, qfeats = [], []

        embeds = []
        embeds_1, embeds_2, embeds_3, embeds_4, embeds_5 = [], [], [], [], []

        # Determine the number of images based on modalities count
        num_modalities = len(modalities)

        for batch in loader:
            pid = batch[0]
            imgs = [img.to(device) for img in batch[1:1 + num_modalities]]

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
                        text_feat = model.encode_text_embeds(imgs[0])[0].float()
                        embeds.append(text_feat.cpu())
                    else:
                        if modalities[0] == 'sk':
                            img_feat = model.encode_sk_embeds(imgs[0])[:, 0, :].float()

                        elif modalities[0] == 'cp':
                            img_feat = model.encode_cp_embeds(imgs[0])[:, 0, :].float()

                        elif modalities[0] == 'nir':
                            img_feat = model.encode_nir_embeds(imgs[0])[:, 0, :].float()

                        embeds.append(img_feat.cpu())

                elif len(modalities) == 2:
                    if 'text' in modalities:
                        text_feat_origin = model.encode_text_embeds(imgs[-1])[0].float()

                        if modalities[1] == 'sk':
                            img_feat = model.encode_sk_embeds(imgs[0])[:, 0, :].float()

                        elif modalities[1] == 'cp':
                            img_feat = model.encode_cp_embeds(imgs[0])[:, 0, :].float()

                        elif modalities[1] == 'nir':
                            img_feat = model.encode_nir_embeds(imgs[0])[:, 0, :].float()

                        embeds.append(img_feat.cpu())
                        embeds_1.append(text_feat_origin.cpu())

                    else:
                        # 3种情况
                        if modalities[0] == 'sk' and modalities[1] == 'cp':
                            img_feat = model.encode_sk_embeds(imgs[0])[:, 0, :].float()
                            img2_feat = model.encode_cp_embeds(imgs[1])[:, 0, :].float()
                        elif modalities[0] == 'sk' and modalities[1] == 'nir':
                            img_feat = model.encode_sk_embeds(imgs[0])[:, 0, :].float()
                            img2_feat = model.encode_nir_embeds(imgs[1])[:, 0, :].float()
                        elif modalities[0] == 'cp' and modalities[1] == 'nir':
                            img_feat = model.encode_cp_embeds(imgs[0])[:, 0, :].float()
                            img2_feat = model.encode_nir_embeds(imgs[1])[:, 0, :].float()
                        embeds.append(img_feat.cpu())
                        embeds_1.append(img2_feat.cpu())

                elif len(modalities) == 3:
                    if 'text' in modalities:
                        text_feat_origin = model.encode_text_embeds(imgs[-1])[0].float()

                        if modalities[1] == 'cp' and modalities[2] == 'sk':
                            img_feat = model.encode_cp_embeds(imgs[0])[:, 0, :].float()
                            img2_feat = model.encode_sk_embeds(imgs[1])[:, 0, :].float()

                        elif modalities[1] == 'cp' and modalities[2] == 'nir':
                            img_feat = model.encode_cp_embeds(imgs[0])[:, 0, :].float()
                            img2_feat = model.encode_nir_embeds(imgs[1])[:, 0, :].float()

                        elif modalities[1] == 'sk' and modalities[2] == 'nir':
                            img_feat = model.encode_sk_embeds(imgs[0])[:, 0, :].float()
                            img2_feat = model.encode_nir_embeds(imgs[1])[:, 0, :].float()

                        # embeds.append(text_feat.cpu())
                        embeds.append(text_feat_origin.cpu())
                        embeds_1.append(img_feat.cpu())
                        embeds_2.append(img2_feat.cpu())
                    else:
                        if modalities[0] == 'cp' and modalities[1] == 'sk' and modalities[2] == 'nir':
                            img_feat = model.encode_cp_embeds(imgs[0])[:, 0, :].float()
                            img2_feat =  model.encode_sk_embeds(imgs[1])[:, 0, :].float()
                            img3_feat = model.encode_nir_embeds(imgs[2])[:, 0, :].float()
                            embeds.append(img_feat.cpu())
                            embeds_1.append(img2_feat.cpu())
                            embeds_2.append(img3_feat.cpu())
                elif len(modalities) == 4:
                    text_feat_origin = model.encode_text_embeds(imgs[-1])[0].float()

                    img_feat = model.encode_cp_embeds(imgs[0])[:, 0, :].float()
                    img2_feat =  model.encode_sk_embeds(imgs[1])[:, 0, :].float()
                    img3_feat = model.encode_nir_embeds(imgs[2])[:, 0, :].float()

                    # embeds.append(text_feat.cpu())
                    embeds.append(text_feat_origin.cpu())
                    embeds_1.append(img_feat.cpu())
                    embeds_2.append(img2_feat.cpu())
                    embeds_3.append(img3_feat.cpu())

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

        return qids, qfeats, qfeats_1, qfeats_2, qfeats_3, qfeats_4, qfeats_5

    def _evaluate_modality(self, gfeats, gids, model, loader, modalities):
        """Generic evaluation function for any modality combination"""
        # if len(modalities) == 1:
        #     qids, qfeats = self._extract_single_modality_features(model, loader, modalities[0])
        # else:
        #     qids, qfeats = self._extract_multi_modality_features(model, loader, modalities)
        # similarity = qfeats @ gfeats.t()

        qids, qfeats, qfeats_1, qfeats_2, qfeats_3, qfeats_4, qfeats_5 = self._extract_multi_modality_features(model, loader, modalities)

        similarity = qfeats @ gfeats.t()
        # similarity_t = qfeats_t @ gfeats_t.t()
        # similarity = (similarity + similarity_t) / 2
        q_num = 1
        if qfeats_1 is not None:
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

    def eval(self, model):
        model = model.eval()
        device = next(model.parameters()).device

        # Extract gallery features
        gids, gfeats = [], []
        for pid, img in self.gallery_loader:
            img = img.to(device)
            with torch.no_grad():
                img_feat = model.encode_rgb_cls(img)
            gids.append(pid.view(-1).cpu())
            gfeats.append(img_feat.cpu())
        gids = torch.cat(gids, 0)
        gfeats = torch.cat(gfeats, 0)
        gfeats = F.normalize(gfeats, p=2, dim=1)

        eval_results = {}
        modality_combinations = self._get_modality_combinations()

        # Group evaluations by modality count for organized printing
        modality_groups = {
            1: "One Modality Evaluating...",
            2: "Two Modalities Evaluating...",
            3: "Three Modalities Evaluating...",
            4: "Four Modalities Evaluating..."
        }

        current_modality_count = 0

        for task_name, modalities, loader in modality_combinations:
            if loader is None:
                continue

            modality_count = len(modalities)
            if modality_count != current_modality_count:
                current_modality_count = modality_count
                self.logger.info(modality_groups.get(modality_count, ""))

            result = self._evaluate_modality(gfeats, gids, model, loader, modalities)
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
        # one_aver = self.table_average_calculation(table, 0, 4)
        # two_aver = self.table_average_calculation(table, 4, 16)
        # three_aver = self.table_average_calculation(table, 16, 28)
        # four_aver = self.table_average_calculation(table, 28, 32)
        one_aver = self.table_average_calculation(table, 0, 4)
        two_aver = self.table_average_calculation(table, 4, 10)
        three_aver = self.table_average_calculation(table, 10, 14)
        four_aver = self.table_average_calculation(table, 14, 15)

        table.add_row(['ONE_AVER', one_aver[0], one_aver[1], one_aver[2], one_aver[3], one_aver[4]])
        table.add_row(['TWO_AVER', two_aver[0], two_aver[1], two_aver[2], two_aver[3], two_aver[4]])
        table.add_row(['THREE_AVER', three_aver[0], three_aver[1], three_aver[2], three_aver[3], three_aver[4]])
        table.add_row(['FOUR_AVER', four_aver[0], four_aver[1], four_aver[2], four_aver[3], four_aver[4]])

        # Format table
        for field in ["R1", "R5", "R10", "mAP", "mINP"]:
            table.custom_format[field] = lambda f, v: f"{v:.3f}"

        self.logger.info('\n' + str(table))

        return (one_aver[3] + two_aver[3] + three_aver[3] + four_aver[3]) / 4

    def table_average_calculation(self, table, first, last):
        selected_rows = table._rows[first:last]
        column_sums = [0] * (len(table.field_names) - 1)
        num_rows = len(selected_rows)

        for row in selected_rows:
            for i, value in enumerate(row[1:]):
                column_sums[i] += value

        averages = [sum_val / num_rows for sum_val in column_sums]
        return averages