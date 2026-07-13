import torch
from typing import List, Dict, Any

from transformers import AutoProcessor, AutoModelForVision2Seq

from utils.simple_tokenizer import SimpleTokenizer


def _decode_caption_tokens(
    tokens: torch.Tensor,
    tokenizer: SimpleTokenizer,
) -> str:
    """
    将 CLIP BPE token（`caption_ids` / `rgb_caption_ids`）反解成自然语言字符串，
    再交给 Qwen 自己的 tokenizer 重新编码。
    供训练与评估阶段共享使用。
    """
    tokens = tokens.tolist()
    # 去掉 padding 之后的部分，只保留到 <|endoftext|>
    eot_id = tokenizer.encoder.get("<|endoftext|>")
    if eot_id in tokens:
        end_idx = tokens.index(eot_id) + 1
        tokens = tokens[:end_idx]
    text = tokenizer.decode(tokens)
    # 简单清理一下特殊符号
    text = text.replace("<|startoftext|>", "").replace("<|endoftext|>", "")
    return text.strip()


@torch.no_grad()
def encode_messages_with_learnable_token(
    model: AutoModelForVision2Seq,
    processor: AutoProcessor,
    messages_list: List[List[Dict[str, Any]]],
    device: torch.device,
) -> torch.Tensor:
    """
    给定一组对话 messages，返回每条样本对应的可学习 token 向量。
    要求传入的 model 上已经挂有 `learnable_token` 参数。
    供训练与评估阶段共享使用。
    """
    model.eval()
    # 兼容 DDP / Accelerate 包装：真正的 Qwen 模型可能在 model.module 里
    base_model = getattr(model, "module", model)

    inputs = processor.apply_chat_template(
        messages_list,
        tokenize=True,
        add_generation_prompt=False,
        padding=True,
        return_dict=True,
        return_tensors="pt",
    )

    input_ids: torch.Tensor = inputs.pop("input_ids").to(device)
    attention_mask: torch.Tensor = inputs.pop("attention_mask").to(device)
    other_inputs = {k: v.to(device) for k, v in inputs.items()}

    input_embeds = base_model.get_input_embeddings()(input_ids)
    # 确保与模型权重 / learnable_token 使用相同 dtype（如 bfloat16）
    input_embeds = input_embeds.to(base_model.learnable_token.dtype)
    bsz = input_embeds.size(0)

    # 在最前面拼接一个可学习 token
    learn_tok = base_model.learnable_token.view(1, 1, -1).expand(bsz, 1, -1)
    input_embeds = torch.cat([learn_tok, input_embeds], dim=1)

    new_attention_mask = torch.cat(
        [
            torch.ones(
                bsz, 1, dtype=attention_mask.dtype, device=device
            ),
            attention_mask,
        ],
        dim=1,
    )

    outputs = model(
        inputs_embeds=input_embeds,
        attention_mask=new_attention_mask,
        output_hidden_states=True,
        **other_inputs,
    )
    last_hidden = outputs.hidden_states[-1]
    token_feat = last_hidden[:, 0, :]
    return token_feat


