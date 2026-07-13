import logging
from typing import Dict, Any, List, Tuple

import torch
import torch.nn.functional as F
from prettytable import PrettyTable
from tqdm.auto import tqdm

from utils.simple_tokenizer import SimpleTokenizer
from utils.metrics import rank
from utils.mllm_utils import (
    _decode_caption_tokens,
    encode_messages_with_learnable_token,
)


class EvaluatorMLLM:
    """
    使用 Qwen-VL + 可学习 token 向量进行检索评估的 Evaluator。

    思路：
    - 对于 gallery：每张 RGB 图像构造一条对话，取可学习 token 作为 gallery 特征。
    - 对于不同模态组合的 query：将该组合内的所有图像（和可选的文本）一起作为一条对话输入，
      同样取可学习 token 作为查询特征。
    - 使用 `utils.metrics.rank` 计算 CMC / mAP / mINP。
    """

    def __init__(self, args, gallery_loader, get_mAP: bool = True, **query_loaders: Dict[str, Any]):
        self.args = args
        self.gallery_loader = gallery_loader
        self.query_loaders = query_loaders
        self.get_mAP = get_mAP
        # 复用 train_mllm.py 中通过 setup_logger("ORBench_MLLM", ...) 配置的 logger
        # 保证评估时的日志能正确输出到同一日志文件和控制台
        self.logger = logging.getLogger("ORBench_MLLM")

        self.clip_tokenizer = SimpleTokenizer()
        # 与 `model.build_mllm.build_train_prompts_from_batch` 中保持一致的描述前缀
        self.rgb_caption_prefix = "Visible image of a person with natural colors: "
        self.nir_caption_prefix = "Near-infrared image of a person with high reflectance contrast: "
        self.cp_caption_prefix = "Color-pencil drawing of a person with vivid colors: "
        self.sk_caption_prefix = "Sketch image of a person with clean line contours: "

        self.logger.info("Initialized EvaluatorMLLM with MLLM evaluation settings.")

    def _get_modalities_for_task(self, task_name: str, num_items: int) -> List[str]:
        """
        根据 task_name 和当前 DataLoader 的输出结构，返回每个 items[idx] 对应的模态类型。
        注意：这里的顺序必须与 `datasets.build` 中构建各类 QueryDataset 时保持一致。
        """
        # 单模态
        if task_name == "NIR":
            return ["NIR"]
        if task_name == "CP":
            return ["CP"]
        if task_name == "SK":
            return ["SK"]
        if task_name == "TEXT":
            return ["TEXT"]

        # 双模态
        if task_name == "CP+NIR":
            # TwoQueryAllVisionDataset: (cp_img, nir_img)
            return ["CP", "NIR"]
        if task_name == "SK+NIR":
            # TwoQueryAllVisionDataset: (sk_img, nir_img)
            return ["SK", "NIR"]
        if task_name == "SK+CP":
            # TwoQueryAllVisionDataset: (sk_img, cp_img)
            return ["SK", "CP"]
        if task_name == "TEXT+NIR":
            # TwoQueryTextVisionDataset: (nir_img, text)
            return ["NIR", "TEXT"]
        if task_name == "TEXT+CP":
            # TwoQueryTextVisionDataset: (cp_img, text)
            return ["CP", "TEXT"]
        if task_name == "TEXT+SK":
            # TwoQueryTextVisionDataset: (sk_img, text)
            return ["SK", "TEXT"]

        # 三模态
        if task_name == "CP+SK+NIR":
            # ThreeQueryAllVisionDataset: (cp_img, sk_img, nir_img)
            return ["CP", "SK", "NIR"]
        if task_name == "TEXT+CP+NIR":
            # ThreeQueryTextVisionVisionDataset: (cp_img, nir_img, text)
            return ["CP", "NIR", "TEXT"]
        if task_name == "TEXT+SK+NIR":
            # ThreeQueryTextVisionVisionDataset: (sk_img, nir_img, text)
            return ["SK", "NIR", "TEXT"]
        if task_name == "TEXT+CP+SK":
            # ThreeQueryTextVisionVisionDataset: (cp_img, sk_img, text)
            return ["CP", "SK", "TEXT"]

        # 四模态
        if task_name == "TEXT+CP+SK+NIR":
            # FourQueryTextVisionVisionVisionDataset: (cp_img, sk_img, nir_img, text)
            return ["CP", "SK", "NIR", "TEXT"]

        # 默认兜底：全按 TEXT 处理，避免崩溃
        return ["TEXT"] * num_items

    def _build_messages_from_batch(
        self,
        batch: Tuple[torch.Tensor, ...],
        is_gallery: bool = False,
        task_name: str = "",
    ) -> Tuple[torch.Tensor, List[List[Dict[str, Any]]]]:
        """
        将 DataLoader 的一个 batch 转成 Qwen-VL 所需的 messages_list。

        约定：
        - batch[0]: pids
        - 其余元素：形状为 [B, ...] 的张量，dim>=3 认为是 image，dim==1 认为是 text tokens。
        """
        pid_tensor = batch[0]
        items = batch[1:]
        bsz = pid_tensor.size(0)

        messages_list: List[List[Dict[str, Any]]] = []

        # gallery：当前只包含 RGB 图像，统一使用 RGB 前缀，并且前缀放在图像前面
        if is_gallery:
            for i in range(bsz):
                contents: List[Dict[str, Any]] = []
                img_tensor = items[0][i]
                contents.append(
                    {
                        "type": "text",
                        "text": self.rgb_caption_prefix,
                    }
                )
                contents.append({"type": "image", "image": img_tensor})

                messages = [
                    {
                        "role": "user",
                        "content": contents,
                    }
                ]
                messages_list.append(messages)
            return pid_tensor, messages_list

        # 非 gallery：根据 task_name 为每个 items[idx] 分配模态标签
        modalities = self._get_modalities_for_task(task_name, len(items))

        for i in range(bsz):
            contents: List[Dict[str, Any]] = []
            decoded_texts: List[str] = []

            for t, mod in zip(items, modalities):
                if not torch.is_tensor(t):
                    continue
                sample = t[i]

                # 图像模态：先加 caption prefix，再加对应图像
                if sample.dim() >= 3 and mod in {"NIR", "CP", "SK"}:
                    if mod == "NIR":
                        prefix = self.nir_caption_prefix
                    elif mod == "CP":
                        prefix = self.cp_caption_prefix
                    elif mod == "SK":
                        prefix = self.sk_caption_prefix
                    else:
                        prefix = ""

                    if prefix:
                        contents.append({"type": "text", "text": prefix})
                    contents.append({"type": "image", "image": sample})

                # 文本模态：记录下来，稍后统一追加到结尾
                elif sample.dim() == 1 and mod == "TEXT":
                    txt = _decode_caption_tokens(sample, self.clip_tokenizer)
                    if txt:
                        decoded_texts.append(txt)

            # 如果存在 TEXT 模态，把文本放在所有图像之后
            if decoded_texts:
                query_text = " ".join(decoded_texts)
                contents.append(
                    {
                        "type": "text",
                        "text": query_text,
                    }
                )

            messages = [
                {
                    "role": "user",
                    "content": contents,
                }
            ]
            messages_list.append(messages)

        return pid_tensor, messages_list

    def _extract_feats(
        self,
        model,
        processor,
        loader,
        is_gallery: bool,
        task_name: str = "",
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        device = next(model.parameters()).device
        all_ids: List[torch.Tensor] = []
        all_feats: List[torch.Tensor] = []

        model.eval()

        data_iter = tqdm(
            loader,
            desc="Gallery" if is_gallery else f"Query-{task_name}",
            leave=False,
        )
        for batch in data_iter:
            batch = tuple(
                x.to(device) if torch.is_tensor(x) else x
                for x in batch
            )
            pids, messages_list = self._build_messages_from_batch(
                batch,
                is_gallery=is_gallery,
                task_name=task_name,
            )

            feats = encode_messages_with_learnable_token(
                model=model,
                processor=processor,
                messages_list=messages_list,
                device=device,
            )

            all_ids.append(pids.view(-1).cpu())
            all_feats.append(feats.cpu())

        ids = torch.cat(all_ids, 0)
        feats = torch.cat(all_feats, 0)
        feats = F.normalize(feats, p=2, dim=1)
        return ids, feats

    def _get_query_combinations(self) -> List[tuple]:
        """
        与原 `utils.metrics.Evaluator._get_modality_combinations` 对齐的组合列表，
        但这里只需要 task_name 与对应 loader。
        """
        return [
            # 单模态
            ("NIR", self.query_loaders.get("nir_query_loader")),
            ("CP", self.query_loaders.get("cp_query_loader")),
            ("SK", self.query_loaders.get("sk_query_loader")),
            ("TEXT", self.query_loaders.get("text_query_loader")),
            # 双模态
            ("CP+NIR", self.query_loaders.get("cp_nir_query_loader")),
            ("SK+NIR", self.query_loaders.get("sk_nir_query_loader")),
            ("TEXT+NIR", self.query_loaders.get("text_nir_query_loader")),
            ("SK+CP", self.query_loaders.get("sk_cp_query_loader")),
            ("TEXT+CP", self.query_loaders.get("text_cp_query_loader")),
            ("TEXT+SK", self.query_loaders.get("text_sk_query_loader")),
            # 三模态
            ("CP+SK+NIR", self.query_loaders.get("cp_sk_nir_query_loader")),
            ("TEXT+CP+NIR", self.query_loaders.get("text_cp_nir_query_loader")),
            ("TEXT+SK+NIR", self.query_loaders.get("text_sk_nir_query_loader")),
            ("TEXT+CP+SK", self.query_loaders.get("text_cp_sk_query_loader")),
            # 四模态
            ("TEXT+CP+SK+NIR", self.query_loaders.get("text_cp_sk_nir_query_loader")),
        ]

    def eval(self, model, processor):
        """
        使用可学习 token 特征做检索评估。
        """
        device = next(model.parameters()).device
        self.logger.info(f"Evaluating MLLM on device: {device}")

        # 1. 提取 gallery 特征
        gids, gfeats = self._extract_feats(
            model,
            processor,
            self.gallery_loader,
            is_gallery=True,
            task_name="GALLERY",
        )

        eval_results: Dict[str, Any] = {}
        query_combos = self._get_query_combinations()

        modality_groups = {
            1: "One Modality Evaluating (MLLM)...",
            2: "Two Modalities Evaluating (MLLM)...",
            3: "Three Modalities Evaluating (MLLM)...",
            4: "Four Modalities Evaluating (MLLM)...",
        }
        current_modality_count = 0

        for task_name, loader in query_combos:
            if loader is None:
                continue

            modality_count = task_name.count("+") + 1
            if modality_count != current_modality_count:
                current_modality_count = modality_count
                self.logger.info(modality_groups.get(modality_count, ""))

            qids, qfeats = self._extract_feats(
                model,
                processor,
                loader,
                is_gallery=False,
                task_name=task_name,
            )

            similarity = qfeats @ gfeats.t()

            t2i_cmc, t2i_mAP, t2i_mINP, _ = rank(
                similarity=similarity,
                q_pids=qids,
                g_pids=gids,
                max_rank=10,
                get_mAP=self.get_mAP,
            )

            t2i_cmc = t2i_cmc.cpu().numpy()
            t2i_mAP = t2i_mAP.cpu().numpy()
            t2i_mINP = t2i_mINP.cpu().numpy()

            result = (
                float(t2i_cmc[0]),
                float(t2i_cmc[4]),
                float(t2i_cmc[9]),
                float(t2i_mAP),
                float(t2i_mINP),
            )
            eval_results[task_name] = result

            self.logger.info(
                f"{task_name}: R1={result[0]:.3f}, R5={result[1]:.3f}, "
                f"R10={result[2]:.3f}, mAP={result[3]:.3f}, mINP={result[4]:.3f}"
            )

        # 汇总表格
        table = PrettyTable(["task", "R1", "R5", "R10", "mAP", "mINP"])
        ordered_names = [name for name, _ in query_combos]
        for task_name in ordered_names:
            if task_name in eval_results:
                r = eval_results[task_name]
                table.add_row([task_name, r[0], r[1], r[2], r[3], r[4]])

        def _avg_rows(rows: List[List[float]]) -> List[float]:
            if not rows:
                return [0.0] * 5
            cols = list(zip(*rows))
            return [sum(c) / len(c) for c in cols]

        # 计算各模态数目的平均
        one_rows = table._rows[0:4]
        two_rows = table._rows[4:10]
        three_rows = table._rows[10:14]
        four_rows = table._rows[14:15]

        one_aver = _avg_rows(one_rows)
        two_aver = _avg_rows(two_rows)
        three_aver = _avg_rows(three_rows)
        four_aver = _avg_rows(four_rows)

        table.add_row(["ONE_AVER", *one_aver])
        table.add_row(["TWO_AVER", *two_aver])
        table.add_row(["THREE_AVER", *three_aver])
        table.add_row(["FOUR_AVER", *four_aver])

        for field in ["R1", "R5", "R10", "mAP", "mINP"]:
            table.custom_format[field] = lambda f, v: f"{float(v):.3f}"

        self.logger.info("\n" + str(table))

        # 返回四类任务 mAP 的平均值，便于外部选择最佳模型
        return (one_aver[3] + two_aver[3] + three_aver[3] + four_aver[3]) / 4.0


