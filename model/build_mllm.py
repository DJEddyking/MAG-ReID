"""
使用 Qwen3-VL-2B 进行 LoRA 微调的示例脚本。

功能：
1. 从 `datasets.build.build_dataloader` 中读取 ORBench 训练集（与原始 CLIP2ReID 相同的数据管线）。
2. 只使用 RGB 图像 (`rgbs`) + 文本描述（从 `caption_ids` 或 `rgb_caption_ids` 反解成字符串）作为监督信号。
3. 基于 HuggingFace Transformers + PEFT 对 Qwen3-VL-2B 做 LoRA 微调（因环境和模型版本不同，细节可按需微调）。

注意：
- 脚本默认只做一个非常基础的 caption 生成任务示例，方便后续根据需求改造。
- 运行前请确保已安装 `transformers>=4.40`, `peft`, `accelerate`，并能从本地或 HuggingFace 加载 Qwen3-VL-2B 模型。
"""

import os
import math
import random
import itertools
import logging
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Sequence

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from accelerate import Accelerator
from transformers import (
    AutoProcessor,
    AutoModelForVision2Seq,
    get_cosine_schedule_with_warmup,
)
from peft import LoraConfig, get_peft_model, TaskType
from tqdm.auto import tqdm

from datasets import build_dataloader
from utils.options import get_args
from utils.simple_tokenizer import SimpleTokenizer
from utils.mllm_utils import _decode_caption_tokens
from utils.metrics_mllm import EvaluatorMLLM

# ========================
# 配置区（可根据需要修改）
# ========================

QWEN_VL_MODEL_NAME = os.environ.get(
    "QWEN_VL_MODEL_NAME",
    # 默认使用你本地已经下载好的 Qwen3-VL-2B-Instruct 权重路径
    "/home/dj_2025/.cache/modelscope/hub/models/Qwen/Qwen3-VL-2B-Instruct",
)


@dataclass
class QwenVLLoRAConfig:
    """LoRA 超参数配置，可按需调整。"""

    r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    learning_rate: float = 1e-4
    weight_decay: float = 0.0
    warmup_ratio: float = 0.03
    max_grad_norm: float = 1.0
    max_epochs: Optional[int] = None  # 若为 None 则使用 args.num_epoch


def _get_combination_chosen() -> List[List[int]]:
    """
    复用训练时的组合策略：
    4 个开关依次表示 [SK, CP, NIR, TEXT] 是否启用（RGB 始终存在）。
    每个 epoch 覆盖 15 种非空组合，并随机打乱顺序。
    """
    combs: List[List[int]] = []
    for num_none in range(1, 5):
        for none_positions in itertools.combinations(range(4), num_none):
            tmp = [1] * 4
            for i in range(4):
                if i not in none_positions:
                    tmp[i] = 0
            combs.append(tmp)
    epoch_combos = list(combs)
    random.shuffle(epoch_combos)
    return epoch_combos


def build_qwen_vl_lora_model(
    model_name: str = QWEN_VL_MODEL_NAME,
    device_dtype: torch.dtype = torch.bfloat16,
) -> (AutoModelForVision2Seq, AutoProcessor):
    """
    加载 Qwen-VL 模型与处理器，并注入 LoRA 适配层。
    """
    processor = AutoProcessor.from_pretrained(
        model_name, trust_remote_code=True
    )

    model = AutoModelForVision2Seq.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype=device_dtype,
        device_map=None,  # 交给 Accelerate 管理
    )

    # 只在注意力和 FFN 层上插 LoRA，一般够用
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model, processor


def build_train_prompts_from_batch(
    batch: Dict[str, Any],
    clip_tokenizer: SimpleTokenizer,
    use_prompt: bool,
    combs_select: Sequence[int],
) -> List[List[Dict[str, Any]]]:
    """
    将一个 batch 转成 Qwen-VL chat messages 列表。
    这里只使用 RGB 图像和对应描述做 caption 生成任务。
    """
    captions_tensor: Optional[torch.Tensor] = None
    if use_prompt and "rgb_caption_ids" in batch:
        captions_tensor = batch["rgb_caption_ids"]
    elif (not use_prompt) and "caption_ids" in batch:
        captions_tensor = batch["caption_ids"]

    rgb_images = batch["rgbs"]
    sk_images = batch.get("sks", None)
    cp_images = batch.get("cps", None)
    nir_images = batch.get("nirs", None)

    rgb_caption_prefix = "Visible image of a person with natural colors: "
    nir_caption_prefix = "Near-infrared image of a person with high reflectance contrast: "
    cp_caption_prefix = "Color-pencil drawing of a person with vivid colors: "
    sk_caption_prefix = "Sketch image of a person with clean line contours: "

    use_sk, use_cp, use_nir, use_text = [
        bool(int(x)) for x in combs_select
    ]

    messages_list: List[Dict[str, Any]] = []

    bsz = rgb_images.size(0)
    for i in range(bsz):
        # 构造多模态图像列表：RGB 一定存在，其它模态按 combs_select 选择
        contents: List[Dict[str, Any]] = []
        contents.append({"type": "text", "text": rgb_caption_prefix})
        contents.append({"type": "image", "image": rgb_images[i]})
        # contents.append({"type": "text", "text": rgb_caption_prefix})
        if use_sk and sk_images is not None:
            contents.append({"type": "text", "text": sk_caption_prefix})
            contents.append({"type": "image", "image": sk_images[i]})
            
        if use_cp and cp_images is not None:
            contents.append({"type": "text", "text": cp_caption_prefix})
            contents.append({"type": "image", "image": cp_images[i]})
            
        if use_nir and nir_images is not None:
            contents.append({"type": "text", "text": nir_caption_prefix})
            contents.append({"type": "image", "image": nir_images[i]})
            

        # # user 指令统一使用英文
        # contents.append(
        #     {
        #         "type": "text",
        #         "text": ,
        #     }
        # )

        messages: List[Dict[str, Any]] = [
            {
                "role": "user",
                "content": contents,
            }
        ]

        # 如果当前组合包含 TEXT，则把 caption 当作 assistant 的监督标签
        if use_text and captions_tensor is not None:
            caption_text = _decode_caption_tokens(
                captions_tensor[i], clip_tokenizer
            )
            if not caption_text:
                caption_text = "A person."
            messages.append(
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "text",
                            "text": caption_text,
                        }
                    ],
                }
            )

        messages_list.append(messages)
    return messages_list


def train_qwen_vl_lora(args: Optional[Any] = None):
    """
    主训练入口：
    - 复用原项目的 `get_args` 和 `build_dataloader` 读取 ORBench；
    - 对 Qwen-VL 做简单的 LoRA SFT，并按设定 epoch 进行 MLLM 检测评估。
    """
    if args is None:
        args = get_args()

    logger = logging.getLogger("ORBench_MLLM")

    # 控制 HuggingFace tokenizers 的并行，避免多进程下死锁警告
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    # 多卡 / 单卡兼容：
    # - 若外部已通过 torch.distributed 初始化多卡（train_mllm.py 中设置 args.distributed），
    #   则不再改动 CUDA_VISIBLE_DEVICES，由各进程的 local_rank 控制当前 GPU。
    # - 若为单卡场景，则仍然支持通过 args.gpu_id 指定具体 GPU。
    if not getattr(args, "distributed", False) and getattr(args, "gpu_id", None) is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)

    # 解析 eval_period，避免字符串导致求模报错
    try:
        eval_period = int(getattr(args, "eval_period", 1))
    except (TypeError, ValueError):
        logger.warning(
            f"Invalid eval_period={getattr(args, 'eval_period', None)}, fallback to 1."
        )
        eval_period = 1

    # Accelerate 会自动读取已有的 torch.distributed 环境（若已通过 torchrun / torch.distributed 初始化），
    # 在多卡场景下会为每个进程分配对应 GPU，并对 DataLoader 做分布式采样封装。
    accelerator = Accelerator()
    device = accelerator.device

    # 构建训练 + 测试数据加载器
    dataloader_tuple = build_dataloader(args)
    (
        train_loader,
        gallery_loader,
        nir_query_loader,
        cp_query_loader,
        sk_query_loader,
        text_query_loader,
        cp_nir_query_loader,
        sk_nir_query_loader,
        text_nir_query_loader,
        sk_cp_query_loader,
        text_cp_query_loader,
        text_sk_query_loader,
        cp_sk_nir_query_loader,
        text_cp_nir_query_loader,
        text_sk_nir_query_loader,
        text_cp_sk_query_loader,
        text_cp_sk_nir_query_loader,
        num_classes,
    ) = dataloader_tuple

    # 构建 Qwen-VL + LoRA
    model, processor = build_qwen_vl_lora_model()
    clip_tokenizer = SimpleTokenizer()

    # 在 Qwen 模型上增加一个可学习 token 和分类头
    hidden_size = getattr(
        model.config,
        "hidden_size",
        getattr(getattr(model.config, "text_config", model.config), "hidden_size"),
    )
    param_dtype = next(model.parameters()).dtype
    model.learnable_token = nn.Parameter(torch.zeros(hidden_size, dtype=param_dtype))
    nn.init.normal_(model.learnable_token, std=0.02)
    model.classifier = nn.Linear(hidden_size, num_classes)
    model.classifier.to(dtype=param_dtype)
    nn.init.normal_(model.classifier.weight, std=0.02)
    if model.classifier.bias is not None:
        nn.init.zeros_(model.classifier.bias)

    ce_criterion = nn.CrossEntropyLoss()

    # 优化器与 scheduler
    cfg = QwenVLLoRAConfig()
    num_epochs = int(cfg.max_epochs or args.num_epoch)
    total_steps = num_epochs * math.ceil(
        len(train_loader.dataset) / (args.batch_size)
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
    )
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(cfg.warmup_ratio * total_steps),
        num_training_steps=total_steps,
    )

    (
        model,
        optimizer,
        train_loader,
        scheduler,
    ) = accelerator.prepare(model, optimizer, train_loader, scheduler)

    model.train()

    # 组合选择（与原训练保持一致的随机覆盖策略）
    epoch_combos = _get_combination_chosen()
    combo_ptr = 0

    best_mAP = 0.0

    for epoch in range(num_epochs):
        epoch_desc = f"Epoch {epoch+1}/{num_epochs}"
        data_iter = tqdm(
            train_loader,
            desc=epoch_desc,
            disable=not accelerator.is_main_process,
        )
        for step, batch in enumerate(data_iter):
            # 原项目将 batch 中的张量都放到指定 GPU；用 Accelerate 后这里直接 to(device)
            for k, v in batch.items():
                if torch.is_tensor(v):
                    batch[k] = v.to(device)

            # 轮转选择当前 step 的模态组合（[SK, CP, NIR, TEXT]）
            if combo_ptr >= len(epoch_combos):
                epoch_combos = _get_combination_chosen()
                combo_ptr = 0
            combs_select = epoch_combos[combo_ptr]
            combo_ptr += 1

            messages_list = build_train_prompts_from_batch(
                batch,
                clip_tokenizer,
                use_prompt=args.use_prompt,
                combs_select=combs_select,
            )

            # Qwen-VL 支持批处理 messages：传入一个 list[list[dict]]
            inputs = processor.apply_chat_template(
                messages_list,
                tokenize=True,
                add_generation_prompt=False,
                padding=True,
                return_dict=True,
                return_tensors="pt",
            )

            # 拆出 input_ids 和 attention_mask，改用 inputs_embeds 注入自定义 token
            input_ids: torch.Tensor = inputs.pop("input_ids").to(device)
            attention_mask: torch.Tensor = inputs.pop("attention_mask").to(device)
            other_inputs = {k: v.to(device) for k, v in inputs.items()}

            # 计算原始 token embedding
            # 注意：在多卡场景下 model 可能被 DDP / Accelerate 包装，直接访问
            # get_input_embeddings / learnable_token 可能会失败，这里使用 unwrap_model
            # 拿到底层 Qwen 模型。
            base_model = accelerator.unwrap_model(model)
            input_embeds = base_model.get_input_embeddings()(input_ids)
            # 确保与模型权重 / learnable_token 使用相同 dtype（如 bfloat16）
            input_embeds = input_embeds.to(base_model.learnable_token.dtype)
            bsz, seq_len, _ = input_embeds.shape

            # 在最前面拼接一个可学习 token
            learn_tok = base_model.learnable_token.view(1, 1, -1).expand(bsz, 1, -1)
            input_embeds = torch.cat([learn_tok, input_embeds], dim=1)

            # 扩展 attention mask，为新 token 设置可见性
            new_attention_mask = torch.cat(
                [
                    torch.ones(
                        bsz, 1, dtype=attention_mask.dtype, device=device
                    ),
                    attention_mask,
                ],
                dim=1,
            )
            # 只做前向传播拿隐藏状态（不再使用 LM loss），用于 ReID 表征学习
            outputs = model(
                inputs_embeds=input_embeds,
                attention_mask=new_attention_mask,
                output_hidden_states=True,
                **other_inputs,
            )
            last_hidden = outputs.hidden_states[-1]  # [bsz, seq_len+1, hidden]
            token_feat = last_hidden[:, 0, :]  # 可学习 token 的最终特征

            # 分类损失（ID loss）
            pids = batch["pids"].to(device)
            # 同样通过底层模型访问 classifier，避免 DDP 封装层无此属性
            cls_logits = base_model.classifier(token_feat)
            cls_loss = ce_criterion(cls_logits, pids)

            # 目前仅使用分类损失进行 ReID 监督（如需对比学习，可在此处加入 SupConLoss 等）
            loss = cls_loss

            accelerator.backward(loss)
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), cfg.max_grad_norm
            )
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            if accelerator.is_main_process:
                # tqdm 动态展示当前 loss
                data_iter.set_postfix(
                    {
                        "loss": f"{loss.item():.4f}",
                        "cls": f"{cls_loss.item():.4f}",
                    }
                )
                if (step + 1) % 50 == 0:
                    logger.info(
                        f"[Epoch {epoch+1}/{num_epochs}] "
                        f"Step {step+1}/{len(train_loader)} "
                        f"Loss {loss.item():.4f} | CLS {cls_loss.item():.4f}"
                    )

        # 按照 eval_period 进行 MLLM 检测评估
        if accelerator.is_main_process and ((epoch + 1) % eval_period == 0):
            logger.info(f"Validation Results (MLLM) - Epoch: {epoch + 1}")
            eval_model = accelerator.unwrap_model(model)
            eval_model.eval()
            eval_model.to(device)

            evaluator = EvaluatorMLLM(
                args=args,
                gallery_loader=gallery_loader,
                get_mAP=True,
                nir_query_loader=nir_query_loader,
                cp_query_loader=cp_query_loader,
                sk_query_loader=sk_query_loader,
                text_query_loader=text_query_loader,
                cp_nir_query_loader=cp_nir_query_loader,
                sk_nir_query_loader=sk_nir_query_loader,
                text_nir_query_loader=text_nir_query_loader,
                sk_cp_query_loader=sk_cp_query_loader,
                text_cp_query_loader=text_cp_query_loader,
                text_sk_query_loader=text_sk_query_loader,
                cp_sk_nir_query_loader=cp_sk_nir_query_loader,
                text_cp_nir_query_loader=text_cp_nir_query_loader,
                text_sk_nir_query_loader=text_sk_nir_query_loader,
                text_cp_sk_query_loader=text_cp_sk_query_loader,
                text_cp_sk_nir_query_loader=text_cp_sk_nir_query_loader,
            )

            avg_mAP = evaluator.eval(eval_model, processor)
            logger.info(f"Average mAP over modality groups (MLLM): {avg_mAP:.4f}")

            if avg_mAP > best_mAP:
                best_mAP = avg_mAP
                logger.info(f"New best average mAP: {best_mAP:.4f} at epoch {epoch + 1}")

            torch.cuda.empty_cache()

        # 每个 epoch 末尾简单做一次保存（只保存 LoRA 权重和头部）
        if accelerator.is_main_process:
            save_dir = os.path.join(
                args.output_dir, f"qwen_vl_lora_epoch_{epoch+1}"
            )
            os.makedirs(save_dir, exist_ok=True)
            accelerator.unwrap_model(model).save_pretrained(save_dir)
            processor.save_pretrained(save_dir)


if __name__ == "__main__":
    train_qwen_vl_lora()
