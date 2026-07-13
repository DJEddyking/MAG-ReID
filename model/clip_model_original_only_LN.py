""" CLIP Model
Adapted from https://github.com/openai/CLIP. Originally MIT License, Copyright (c) 2021 OpenAI.
"""
from collections import OrderedDict
import logging
import math
import os
from typing import List, Tuple, Union
import hashlib
import urllib
from tqdm import tqdm
import warnings
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, Tensor
from torch.nn.init import constant_, xavier_normal_, xavier_uniform_
from typing import Optional, Tuple
from torch.nn.modules.activation import _is_make_fx_tracing, _check_arg_device, _arg_requires_grad

from .utils import InputImageType


logger = logging.getLogger("CLIP2ReID.model")

_MODELS = {
    "RN50": "https://openaipublic.azureedge.net/clip/models/afeb0e10f9e5a86da6080e35cf09123aca3b358a0c3e3b6c78a7b63bc04b6762/RN50.pt",
    "RN101": "https://openaipublic.azureedge.net/clip/models/8fa8567bab74a42d41c5915025a8e4538c3bdbe8804a470a72f30b0d94fab599/RN101.pt",
    "RN50x4": "https://openaipublic.azureedge.net/clip/models/7e526bd135e493cef0776de27d5f42653e6b4c8bf9e0f653bb11773263205fdd/RN50x4.pt",
    "RN50x16": "https://openaipublic.azureedge.net/clip/models/52378b407f34354e150460fe41077663dd5b39c54cd0bfd2b27167a4a06ec9aa/RN50x16.pt",
    "RN50x64": "https://openaipublic.azureedge.net/clip/models/be1cfb55d75a9666199fb2206c106743da0f6468c9d327f3e0d0a543a9919d9c/RN50x64.pt",
    "ViT-B/32": "https://openaipublic.azureedge.net/clip/models/40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/ViT-B-32.pt",
    "ViT-B/16": "https://openaipublic.azureedge.net/clip/models/5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f/ViT-B-16.pt",
    "ViT-L/14": "https://openaipublic.azureedge.net/clip/models/b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836/ViT-L-14.pt",
}

def available_models() -> List[str]:
    """Returns the names of available CLIP models"""
    return list(_MODELS.keys())

def _download(url: str, root: str):
    os.makedirs(root, exist_ok=True)
    filename = os.path.basename(url)

    expected_sha256 = url.split("/")[-2]
    download_target = os.path.join(root, filename)

    if os.path.exists(download_target) and not os.path.isfile(download_target):
        raise RuntimeError(f"{download_target} exists and is not a regular file")

    if os.path.isfile(download_target):
        if hashlib.sha256(open(download_target, "rb").read()).hexdigest() == expected_sha256:
            return download_target
        else:
            warnings.warn(f"{download_target} exists, but the SHA256 checksum does not match; re-downloading the file")

    with urllib.request.urlopen(url) as source, open(download_target, "wb") as output:
        with tqdm(total=int(source.info().get("Content-Length")), ncols=80, unit='iB', unit_scale=True, unit_divisor=1024) as loop:
            while True:
                buffer = source.read(8192)
                if not buffer:
                    break

                output.write(buffer)
                loop.update(len(buffer))

    if hashlib.sha256(open(download_target, "rb").read()).hexdigest() != expected_sha256:
        raise RuntimeError(f"Model has been downloaded but the SHA256 checksum does not not match")

    return download_target



class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1):
        super().__init__()

        # all conv layers have stride 1. an avgpool is performed after the second convolution when stride > 1
        self.conv1 = nn.Conv2d(inplanes, planes, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)

        self.conv2 = nn.Conv2d(planes, planes, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.avgpool = nn.AvgPool2d(stride) if stride > 1 else nn.Identity()

        self.conv3 = nn.Conv2d(planes, planes * self.expansion, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = None
        self.stride = stride

        if stride > 1 or inplanes != planes * Bottleneck.expansion:
            # downsampling layer is prepended with an avgpool, and the subsequent convolution has stride 1
            self.downsample = nn.Sequential(OrderedDict([
                ("-1", nn.AvgPool2d(stride)),
                ("0", nn.Conv2d(inplanes, planes * self.expansion, 1, stride=1, bias=False)),
                ("1", nn.BatchNorm2d(planes * self.expansion))
            ]))

    def forward(self, x: torch.Tensor):
        identity = x

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.avgpool(out)
        out = self.bn3(self.conv3(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out


class AttentionPool2d(nn.Module):
    def __init__(self, spacial_dim: int, embed_dim: int, num_heads: int, output_dim: int = None):
        super().__init__()
        # self.positional_embedding = nn.Parameter(torch.randn(spacial_dim ** 2 + 1, embed_dim) / embed_dim ** 0.5)
        self.positional_embedding = nn.Parameter(torch.randn((spacial_dim[0] * spacial_dim[1]) + 1, embed_dim)/ embed_dim ** 0.5)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.c_proj = nn.Linear(embed_dim, output_dim or embed_dim)
        self.num_heads = num_heads

    def forward(self, x):
        x = x.reshape(x.shape[0], x.shape[1], x.shape[2] * x.shape[3]).permute(2, 0, 1)  # NCHW -> (HW)NC
        x = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0)  # (HW+1)NC
        x = x + self.positional_embedding[:, None, :].to(x.dtype)  # (HW+1)NC
        x, _ = F.multi_head_attention_forward(
            query=x, key=x, value=x,
            embed_dim_to_check=x.shape[-1],
            num_heads=self.num_heads,
            q_proj_weight=self.q_proj.weight,
            k_proj_weight=self.k_proj.weight,
            v_proj_weight=self.v_proj.weight,
            in_proj_weight=None,
            in_proj_bias=torch.cat([self.q_proj.bias, self.k_proj.bias, self.v_proj.bias]),
            bias_k=None,
            bias_v=None,
            add_zero_attn=False,
            dropout_p=0,
            out_proj_weight=self.c_proj.weight,
            out_proj_bias=self.c_proj.bias,
            use_separate_proj_weight=True,
            training=self.training,
            need_weights=False
        )

        return x[0]


class ModifiedResNet(nn.Module):
    """
    A ResNet class that is similar to torchvision's but contains the following changes:
    - There are now 3 "stem" convolutions as opposed to 1, with an average pool instead of a max pool.
    - Performs anti-aliasing strided convolutions, where an avgpool is prepended to convolutions with stride > 1
    - The final pooling layer is a QKV attention instead of an average pool
    """

    def __init__(self, layers, output_dim, heads, input_resolution=224, width=64):
        super().__init__()
        self.output_dim = output_dim
        self.input_resolution = input_resolution

        # the 3-layer stem
        self.conv1 = nn.Conv2d(3, width // 2, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width // 2)
        self.conv2 = nn.Conv2d(width // 2, width // 2, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(width // 2)
        self.conv3 = nn.Conv2d(width // 2, width, kernel_size=3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(width)
        self.avgpool = nn.AvgPool2d(2)
        self.relu = nn.ReLU(inplace=True)

        # residual layers
        self._inplanes = width  # this is a *mutable* variable used during construction
        self.layer1 = self._make_layer(width, layers[0])
        self.layer2 = self._make_layer(width * 2, layers[1], stride=2)
        self.layer3 = self._make_layer(width * 4, layers[2], stride=2)
        self.layer4 = self._make_layer(width * 8, layers[3], stride=2)

        embed_dim = width * 32  # the ResNet feature dimension
        spacial_dim = (
            input_resolution[0] // 32,
            input_resolution[1] // 32,
        )
        self.attnpool = AttentionPool2d(spacial_dim, embed_dim, heads, output_dim)

    def _make_layer(self, planes, blocks, stride=1):
        layers = [Bottleneck(self._inplanes, planes, stride)]

        self._inplanes = planes * Bottleneck.expansion
        for _ in range(1, blocks):
            layers.append(Bottleneck(self._inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        def stem(x):
            for conv, bn in [(self.conv1, self.bn1), (self.conv2, self.bn2), (self.conv3, self.bn3)]:
                x = self.relu(bn(conv(x)))
            x = self.avgpool(x)
            return x

        x = x.type(self.conv1.weight.dtype)
        x = stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.attnpool(x)

        return x


class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class Adapter(nn.Module):
    def __init__(self,
                #  config=None,
                 d_model=None,
                 bottleneck=None,
                 dropout=0.0,
                 init_option="bert",
                 adapter_scalar="1.0",
                 adapter_layernorm_option="in"):
        super().__init__()
        self.n_embd = d_model # config.d_model if d_model is None else d_model
        self.down_size = bottleneck # config.attn_bn if bottleneck is None else bottleneck, 下采样的维度

        #_before
        self.adapter_layernorm_option = adapter_layernorm_option

        self.adapter_layer_norm_before = None
        if adapter_layernorm_option == "in" or adapter_layernorm_option == "out":
            self.adapter_layer_norm_before = nn.LayerNorm(self.n_embd)

        if adapter_scalar == "learnable_scalar":
            self.scale = nn.Parameter(torch.ones(1))
        else:
            self.scale = float(adapter_scalar)

        self.down_proj = nn.Linear(self.n_embd, self.down_size)
        self.non_linear_func = QuickGELU() # nn.ReLU()
        self.up_proj = nn.Linear(self.down_size, self.n_embd)

        self.dropout = dropout
        if init_option == "bert":
            raise NotImplementedError
        elif init_option == "lora":  ########## 本质上就是 Lora ???
            with torch.no_grad():
                nn.init.kaiming_uniform_(self.down_proj.weight,  a=0, mode='fan_in', nonlinearity='relu')#, a=math.sqrt(5))  ##### 
                nn.init.zeros_(self.up_proj.weight)
                nn.init.zeros_(self.down_proj.bias)
                nn.init.zeros_(self.up_proj.bias)

    def forward(self, x, add_residual=True, residual=None):
        residual = x if residual is None else residual
        if self.adapter_layernorm_option == 'in':
            x = self.adapter_layer_norm_before(x)

        down = self.down_proj(x)
        down = self.non_linear_func(down)
        down = nn.functional.dropout(down, p=self.dropout, training=self.training)  ###
        up = self.up_proj(down)

        up = up * self.scale

        if self.adapter_layernorm_option == 'out':
            up = self.adapter_layer_norm_before(up)

        if add_residual:
            output = up + residual
        else:
            output = up

        return output


class LoRaExpertQKV(nn.Module):
    """
     y = W·x + (alpha/r) * B·A·x
    """
    def __init__(self,
                in_features, 
                out_features,
                 rank=4,
                 alpha=32,
                 weight=None,  ### 全连接层自带的权重和偏置
                 bias=None,
                 dropout=0.0):
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.weight = weight
        self.bias = bias

        self.lora_A = nn.Parameter(torch.randn(in_features, rank))
        self.lora_B = nn.Parameter(torch.zeros(rank, out_features))
        
        self.scaling = self.alpha / self.rank
        self.lora_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        with torch.no_grad():
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)

    def forward(self, x):
        # B,N,C比二维高用 einsum or torch manual
        B, N, C = x.shape
        x = x.view(B, N, 3, -1)  # [bs, n, 3, 768]
        o_w = self.weight.view(3, -1, C // 3)  ## [3, 768, 768]
        # 逐个通道求解: original weights
        q = torch.einsum('bny,xy->bnx', x[:, :, 0,:], o_w[0])   # [B, N, 768]
        k = torch.einsum('bny,xy->bnx', x[:, :, 1,:], o_w[1])
        v = torch.einsum('bny,xy->bnx', x[:, :, 2,:], o_w[2])
        base = torch.concat((q,k,v), dim=2)
        if self.bias is not None:
            base = base + self.bias

        # ΔW = lora_B @ lora_A
        lora_A = self.lora_A.view(3, C//3, -1)
        delta_q = torch.einsum('bnx,xr->bnr', self.lora_dropout(x[:, :, 0,:]), lora_A[0])   # [B, N, 768]
        delta_q = torch.einsum('bnr,rx->bnx', delta_q, self.lora_B)
        out_q = q + delta_q * self.scaling

        delta_k = torch.einsum('bnx,xr->bnr', self.lora_dropout(x[:, :, 1,:]), lora_A[1])
        delta_k = torch.einsum('bnr,rx->bnx', delta_k, self.lora_B)
        out_k = k + delta_k * self.scaling

        delta_v = torch.einsum('bnx,xr->bnr', self.lora_dropout(x[:, :, 2,:]), lora_A[2])
        delta_v = torch.einsum('bnr,rx->bnx', delta_v, self.lora_B)
        out_v = v + delta_v * self.scaling

        delta = torch.concat((out_q, out_k, out_v), dim=2)
        return base + delta


class LoRaExpert(nn.Module):
    """
     y = W·x + (alpha/r) * B·A·x
    """
    def __init__(self,
                in_features, 
                out_features,
                 rank=4,
                 alpha=32,
                 weight=None,  ### 全连接层自带的权重和偏置
                 bias=None,
                 dropout=0.0):
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.weight = weight
        self.bias = bias

        # 自带的权重和偏置冻结, 只更新lora
        # self.weight.requires_grad = False
        # self.bias.requires_grad = False

        self.lora_A1 = nn.Parameter(torch.randn(in_features, rank))
        self.lora_B1 = nn.Parameter(torch.zeros(rank, out_features))
        self.lora_A2 = nn.Parameter(torch.randn(in_features, rank))
        self.lora_B2 = nn.Parameter(torch.zeros(rank, out_features))
        self.lora_A3 = nn.Parameter(torch.randn(in_features, rank))
        self.lora_B3 = nn.Parameter(torch.zeros(rank, out_features))
        
        self.scaling = self.alpha / self.rank
        self.lora_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        with torch.no_grad():
            nn.init.kaiming_uniform_(self.lora_A1, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B1)
            nn.init.kaiming_uniform_(self.lora_A2, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B2)
            nn.init.kaiming_uniform_(self.lora_A3, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B3)

    def forward(self, x):
        # B,N,C比二维高用 einsum
        base = torch.einsum('bny,xy->bnx', x, self.weight)   # [B, N, 768]
        if self.bias is not None:
            base = base + self.bias
        # nn.mutliheadattention是对一个query直接乘上3倍的维度, 所以需要分为3个Lora, 代表qkv, 然后拆分为lora ab结果后拼接
        # ΔW = lora_B @ lora_A
        delta1 = torch.einsum('bnx,xr->bnr', self.lora_dropout(x), self.lora_A1)
        delta1 = torch.einsum('bnr,rx->bnx', delta1, self.lora_B1)
        delta2 = torch.einsum('bnx,xr->bnr', self.lora_dropout(x), self.lora_A2)
        delta2 = torch.einsum('bnr,rx->bnx', delta2, self.lora_B2)
        delta3 = torch.einsum('bnx,xr->bnr', self.lora_dropout(x), self.lora_A3)
        delta3 = torch.einsum('bnr,rx->bnx', delta3, self.lora_B3)
        delta = torch.concat((delta1, delta2, delta3), dim=-1)

        return base + delta * self.scaling


class LoRaExpertOut(nn.Module):
    """
     y = W·x + (alpha/r) * B·A·x, 只用于out proj
    """
    def __init__(self,
                in_features, 
                out_features,
                 rank=4,
                 alpha=32,
                 weight=None,  ### 全连接层自带的权重和偏置
                 bias=None,
                 dropout=0.0):
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.weight = weight
        self.bias = bias

        # 自带的权重和偏置冻结, 只更新lora
        # self.weight.requires_grad = False
        # self.bias.requires_grad = False


        self.lora_A = nn.Parameter(torch.randn(in_features, rank))
        self.lora_B = nn.Parameter(torch.zeros(rank, out_features))
        
        self.scaling = self.alpha / self.rank
        self.lora_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        with torch.no_grad():
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)

    def forward(self, x):
        # B,N,C比二维高用 einsum
        base = torch.einsum('bny,xy->bnx', x, self.weight)   # [B, N, 768]
        if self.bias is not None:
            base = base + self.bias
        delta = torch.einsum('bnx,xr->bnr', self.lora_dropout(x), self.lora_A)
        delta = torch.einsum('bnr,rx->bnx', delta, self.lora_B)
        return base + delta * self.scaling
    

class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        # 0722: need_weight改为True, 返回attention maps
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x):
        # ln_1 -> attention -> + x -> ln_2 -> mlp -> + x(这个x是第一次加完后的 !!)
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x
        # x = inputs[0]
        # atten, atten_weight = self.attention(self.ln_1(x))
        # x = x + atten
        # x = x + self.mlp(self.ln_2(x))
        # return [x, atten_weight]
    

class MultiheadAttentionAdapter(nn.MultiheadAttention):
    def __init__(self, embed_dim, num_heads, **kwargs):
        super().__init__(embed_dim, num_heads, **kwargs)
        # factory_kwargs = {'device': device, 'dtype': dtype}
        self.chosen_dict = {
            InputImageType.rgb: 0,
            InputImageType.sketch: 1,
            InputImageType.color_pencil: 2,
            InputImageType.nir: 3
        }
        # 原始attention一共四个参数
        self.in_proj_list = nn.ModuleList(
            LoRaExpert(
                in_features=embed_dim,
                out_features=embed_dim,
                rank=4,
                alpha=4,
                weight=self.in_proj_weight,
                bias=self.in_proj_bias,
                # dropout=0.1
            ).half()
            for _ in range(4)
        )
        self.out_proj_list = nn.ModuleList(
            LoRaExpertOut(
                in_features=embed_dim,
                out_features=embed_dim,
                rank=4,
                alpha=4,
                weight=self.out_proj.weight,
                bias=self.out_proj.bias,
                # dropout=0.1
            ).half()
            for _ in range(4)
        )

        self.scale = self.head_dim ** -0.5

        # self._reset_parameters_all()

    # def _reset_parameters_all(self):
    #     for each_in_proj_weight in self.in_proj_list:
    #         xavier_uniform_(each_in_proj_weight)
    #         xavier_uniform_(each_in_proj_weight)
    #         xavier_uniform_(each_in_proj_weight)

    #     for each_in_proj_bias in self.in_proj_list:
    #         constant_(each_in_proj_bias, 0.)

    #     for each_out_proj_bias in self.out_proj_list:
    #         constant_(each_out_proj_bias.bias, 0.)

    def forward(
            self,
            query: Tensor,
            key: Tensor,
            value: Tensor,
            key_padding_mask: Optional[Tensor] = None,
            need_weights: bool = True,
            attn_mask: Optional[Tensor] = None,
            average_attn_weights: bool = True,
            is_causal : bool = False,
            
            input_img_type: InputImageType = None
            ) -> Tuple[Tensor, Optional[Tensor]]:
        r"""copy from nn.MutiheadAttention
        """
        ### 
        # q,k,v: [N, B, dim]

        # 保持与官方一致的输入格式
        if self.batch_first == False:
            query = query.transpose(0, 1)
            key = key.transpose(0, 1)
            value = value.transpose(0, 1)

        in_proj = self.in_proj_list[self.chosen_dict.get(input_img_type)]
        out_proj = self.out_proj_list[self.chosen_dict.get(input_img_type)]


        for i_k, i_v in self.chosen_dict.items():
            if i_k != input_img_type:
                self.out_proj_list[i_v].requires_grad_(False)
            else:
                self.out_proj_list[i_v].requires_grad_(True)
        # qkv = torch.concat((query, key, value), dim=2).transpose(0, 1)
        B, N, C = query.shape
        # qkv = in_proj(qkv).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        # q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple), [B, head, N, head_dim]
        qkv = in_proj(query)


        # B, N, C = query.shape
        # q = torch.einsum('nbx,xr->nbr', query, self.in_proj_weight.view(3, -1, query.shape[-1])[0].T).transpose(0, 1)
        # q = q + self.in_proj_bias.view(3, -1)[0]
        # q = q.reshape(q.shape[0], q.shape[1], self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        # k = torch.einsum('nbx,xr->nbr', key, self.in_proj_weight.view(3, -1, key.shape[-1])[1].T).transpose(0, 1)
        # k = k + self.in_proj_bias.view(3, -1)[1]
        # k = k.reshape(k.shape[0], k.shape[1], self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        # v = torch.einsum('nbx,xr->nbr', value, self.in_proj_weight.view(3, -1, value.shape[-1])[2].T).transpose(0, 1)
        # v = v + self.in_proj_bias.view(3, -1)[2]
        # v = v.reshape(v.shape[0], v.shape[1], self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        ### nn.multiheadattention本质上只用到了query, 因为输入qkv一样的
        # qkv = F.linear(query, self.in_proj_weight, self.in_proj_bias)
        q, k, v = qkv.chunk(3, dim=-1)

        # reshape 为 (B, head num, N, head dim)
        q = q.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)

        # 缩放: 关键, q先scale
        q = q * self.scale

        # 注意力计算
        attn_scores = torch.matmul(q, k.transpose(-2, -1))

        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = F.dropout(attn_weights, p=self.dropout, training=self.training)

        out = torch.matmul(attn_weights, v)  # (B, H, L, D)
        out = out.transpose(1, 2).contiguous().view(B, N, out_proj.weight.shape[-1])

        # 输出投影
        out = out_proj(out)

        # attn = (q @ k.transpose(-2, -1)) * self.scale
        # attn = attn.softmax(dim=-1)
        # # attn = self.attn_drop(attn)

        # x = (attn @ v).transpose(1, 2).reshape(B, N, C)#, C // 3)
        # # x = out_proj(x)

        # x = torch.einsum('nbx,xr->nbr', x, self.out_proj.weight.T)#.transpose(0, 1)
        # x = x + self.out_proj.bias
        # # x = x.transpose(0, 1)  # 变为[N, B, dim]

        # x = self.proj_drop(x)
        # 加入残差: 不在这里加入, 在线性层加入
        # x = super().forward(query, key, value, key_padding_mask, need_weights, attn_mask, average_attn_weights, is_causal)[0]
        # x = x + original_x

        if self.batch_first == False:
            out = out.transpose(0, 1)

        return out


class Attention(nn.Module):
    """
    自定义self attention
    """
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x
    

class ResidualAttentionBlockWithAdapter(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None, args = None):  # args默认是 None, 防止修改其他调用改类的内容
        super().__init__()

        """
        1.前面几个LN共享, 但是后面的LN不共享, 并且采用adaLN方式
        """
        # 注意：仅保留 LN1/LN2 为非共享，其余层（Attention/MLP）全部共享
        # 使用标准 MultiheadAttention 以确保共享
        self.attn = nn.MultiheadAttention(d_model, n_head)
        # 运行时注入的层索引，由构建器设置
        self.block_id = -1
        
        # 为不同模态创建各自独立的 LN1/LN2，其余权重共享
        self.ln_1 = LayerNorm(d_model)
        self.ln_1_si = LayerNorm(d_model, elementwise_affine=False)
        self.ln_1_ci = LayerNorm(d_model, elementwise_affine=False)
        self.ln_1_ni = LayerNorm(d_model, elementwise_affine=False)
        self.args = args
        # 训练中在 adaLN 与全局仿射之间随机选择的概率（选择全局仿射的概率）
        self.global_affine_prob = getattr(self.args, 'global_affine_prob', 0.5) if self.args is not None else 0.5
        # 仅前 N 层对非 RGB 模态使用来自 RGB 的条件，后续层改为各自 LN 的仿射参数
        self.cond_layers = getattr(self.args, 'cond_layers', 2) if self.args is not None else 2

        ## 第0到第5个transformer block共享LN,
        self.shared_layers = getattr(self.args, 'shared_layers', [0, 6]) if self.args is not None else [0, 6]
        # 一致性正则：全局仿射需与线性映射对齐
        self.adaln_match_loss_weight = getattr(self.args, 'adaln_match_loss_weight', 1.0) if self.args is not None else 1.0
        self.register_buffer("adaln_reg", torch.tensor(0.0))
        self.register_buffer("adaln_reg_count", torch.tensor(0.0))
        # 后续层使用的“各模态自有”仿射 LN
        self.ln_1_si_affine = LayerNorm(d_model)
        self.ln_1_ci_affine = LayerNorm(d_model)
        self.ln_1_ni_affine = LayerNorm(d_model)

        # adaLN: 使用来自 RGB 的条件，分别为各非 RGB 模态学习独立的 gamma/beta 投影
        # self.gamma1_proj = nn.ModuleDict({
        #     "sketch": nn.Linear(d_model, d_model),
        #     "color_pencil": nn.Linear(d_model, d_model),
        #     "nir": nn.Linear(d_model, d_model),
        # })
        # self.beta1_proj = nn.ModuleDict({
        #     "sketch": nn.Linear(d_model, d_model),
        #     "color_pencil": nn.Linear(d_model, d_model),
        #     "nir": nn.Linear(d_model, d_model),
        # })
        # self.gamma2_proj = nn.ModuleDict({
        #     "sketch": nn.Linear(d_model, d_model),
        #     "color_pencil": nn.Linear(d_model, d_model),
        #     "nir": nn.Linear(d_model, d_model),
        # })
        # self.beta2_proj = nn.ModuleDict({
        #     "sketch": nn.Linear(d_model, d_model),
        #     "color_pencil": nn.Linear(d_model, d_model),
        #     "nir": nn.Linear(d_model, d_model),
        # })

        # 采用多模态adaLN策略: 配合gate, score
        # self.adaLN_modulation_proj = nn.ModuleDict({
        #     "sketch": nn.Sequential(
        #         nn.SiLU(),
        #         nn.Linear(d_model, 6 * d_model, bias=True)
        #     ),
        #     "color_pencil": nn.Sequential(
        #         nn.SiLU(),
        #         nn.Linear(d_model, 6 * d_model, bias=True)
        #     ),
        #     "nir": nn.Sequential(
        #         nn.SiLU(),
        #         nn.Linear(d_model, 6 * d_model, bias=True)
        #     ),
        # })
        # # 输入是一个单模态 TODO softmax作为score分数乘上shirt, gate, scale, 然后输出新的scale, shirt, gate
        # self.score_proj = nn.Sequential(
        #     nn.SiLU(),
        #     nn.Linear(d_model, 6 * d_model, bias=True),
        #     nn.Softmax(dim=-1),
        # )


        # 全局可学习仿射参数（每层、每非 RGB 模态独立）
        # self.gamma1_global = nn.ParameterDict({
        #     "sketch": nn.Parameter(torch.ones(d_model)),
        #     "color_pencil": nn.Parameter(torch.ones(d_model)),
        #     "nir": nn.Parameter(torch.ones(d_model)),
        # })
        # self.beta1_global = nn.ParameterDict({
        #     "sketch": nn.Parameter(torch.zeros(d_model)),
        #     "color_pencil": nn.Parameter(torch.zeros(d_model)),
        #     "nir": nn.Parameter(torch.zeros(d_model)),
        # })
        self.gamma2_global = nn.ParameterDict({
            "sketch": nn.Parameter(torch.zeros(d_model)), ## 初始为0, 用来做残差相加
            "color_pencil": nn.Parameter(torch.zeros(d_model)),
            "nir": nn.Parameter(torch.zeros(d_model)),
        })
        self.beta2_global = nn.ParameterDict({
            "sketch": nn.Parameter(torch.zeros(d_model)),
            "color_pencil": nn.Parameter(torch.zeros(d_model)),
            "nir": nn.Parameter(torch.zeros(d_model)),
        })

        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))

        # 如果按照 ( 1 + scale ) * LN(x) + shirt 的方式, 定义若干全局可学习参数
        # self.scale_1_global = nn.ParameterDict(
        #     {
        #         'sketch': nn.Parameter(torch.zeros(d_model)),
        #         'color_pencil': nn.Parameter(torch.zeros(d_model)),
        #         'nir': nn.Parameter(torch.zeros(d_model)),
        #     }
        # )
        # self.scale_2_global = nn.ParameterDict(
        #     {
        #         'sketch': nn.Parameter(torch.zeros(d_model)),
        #         'color_pencil': nn.Parameter(torch.zeros(d_model)),
        #         'nir': nn.Parameter(torch.zeros(d_model)),
        #     }
        # )
        # self.shift_1_global = nn.ParameterDict(
        #     {
        #         'sketch': nn.Parameter(torch.zeros(d_model)),
        #         'color_pencil': nn.Parameter(torch.zeros(d_model)),
        #         'nir': nn.Parameter(torch.zeros(d_model)),
        #     }
        # )
        # self.shift_2_global = nn.ParameterDict(
        #     {
        #         'sketch': nn.Parameter(torch.zeros(d_model)),
        #         'color_pencil': nn.Parameter(torch.zeros(d_model)),
        #         'nir': nn.Parameter(torch.zeros(d_model)),
        #     }
        # )
        # self.gate_1_global = nn.ParameterDict(
        #     {
        #         'sketch': nn.Parameter(torch.ones(d_model)),
        #         'color_pencil': nn.Parameter(torch.ones(d_model)),
        #         'nir': nn.Parameter(torch.ones(d_model)),
        #     }
        # )
        # self.gate_2_global = nn.ParameterDict(
        #     {
        #         'sketch': nn.Parameter(torch.ones(d_model)),
        #         'color_pencil': nn.Parameter(torch.ones(d_model)),
        #         'nir': nn.Parameter(torch.ones(d_model)),
        #     }
        # )


        # dj: 1105 冻结原来 FFN 里面的 linear
        # self.mlp.c_fc.weight.requires_grad = False
        # self.mlp.c_fc.bias.requires_grad = False
        # self.mlp.c_proj.weight.requires_grad = False
        # self.mlp.c_proj.bias.requires_grad = False

        self.ln_2 = LayerNorm(d_model)
        self.ln_2_si = LayerNorm(d_model, elementwise_affine=False)
        self.ln_2_ci = LayerNorm(d_model, elementwise_affine=False)
        self.ln_2_ni = LayerNorm(d_model, elementwise_affine=False)
        self.attn_mask = attn_mask

        self.ln_2_si_affine = LayerNorm(d_model)
        self.ln_2_ci_affine = LayerNorm(d_model)
        self.ln_2_ni_affine = LayerNorm(d_model)

    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        # 使用共享的标准 MultiheadAttention
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    # def forward(self, x: torch.Tensor, input_img_type: InputImageType = None):
    def forward(self, new_x):
        """注意输入, nn.sequential只能单输入; 得到的x维度是 [n, bs, dim]"""
        assert isinstance(new_x, tuple)
        # 兼容三元组输入：(x, input_img_type, cond_holder)
        # cond_holder: 可以是 Tensor[B, D]（旧逻辑）或 List[Optional[Tensor[B,D]]]（分层条件缓存）
        if len(new_x) >= 3:
            x, input_img_type, cond_holder = new_x[0], new_x[1], new_x[2]
        else:
            x, input_img_type = new_x[0], new_x[1]
            cond_holder = None
        # 解析分层条件缓存
        rgb_layer_conds = cond_holder if isinstance(cond_holder, dict) else None
        rgb_cond = cond_holder if (cond_holder is not None and not isinstance(cond_holder, dict)) else None

        # 选择对应模态的 LN1/LN2
        # 是否在本层对非 RGB 模态使用 RGB 条件
        use_cond = (input_img_type != InputImageType.rgb) and (self.block_id is not None) and \
                        self.block_id not in list(range(self.shared_layers[0], self.shared_layers[1]))#(self.block_id < self.cond_layers)

        # if self.block_id in list(range(self.shared_layers[0], self.shared_layers[1])):
        #     # 所有模态共享LN层
        #     ln1 = self.ln_1
        #     ln2 = self.ln_2
        # else:
        ln1 = self.ln_1
        if input_img_type == InputImageType.rgb:
            # ln1 = self.ln_1
            ln2 = self.ln_2
        elif input_img_type == InputImageType.sketch:
            # ln1 = self.ln_1_si if use_cond else self.ln_1_si_affine
            ln2 = self.ln_2_si if use_cond else self.ln_2_si_affine
        elif input_img_type == InputImageType.color_pencil:
            # ln1 = self.ln_1_ci if use_cond else self.ln_1_ci_affine
            ln2 = self.ln_2_ci if use_cond else self.ln_2_ci_affine
        elif input_img_type == InputImageType.nir:
            # ln1 = self.ln_1_ni if use_cond else self.ln_1_ni_affine
            ln2 = self.ln_2_ni if use_cond else self.ln_2_ni_affine
        else:
            # ln1 = self.ln_1
            ln2 = self.ln_2

        # LN1 -> (可选 adaLN 由 RGB 条件调制) -> Attention（共享）
        ln1_out = ln1(x)
        # 计算/读取对应层的 RGB 条件（按层对齐）
        # rgb_cond_block = None
        # if input_img_type == InputImageType.rgb:
        # #     # 从当前层的 LN1 输出聚合得到条件，并写入缓存（若提供）
        #     if rgb_layer_conds is not None and self.block_id is not None and self.block_id >= 0:
        # #         # ln1_out: [L, B, D] -> [B, D]
        #         # rgb_cond_block = ln1_out.mean(dim=0)  # TODO 取平均是否合理 ??? 这个是rgb的ln输出, 取rgb经过ln1之后的特征吗??? 应该是经过ln1之前的?? TODO test
        # #         # 只取cls token
        #         rgb_cond_block = ln1_out[0, :, :]
        #         # rgb_cond_block = ln1_out
        #         rgb_layer_conds['ln_1'][self.block_id] = rgb_cond_block#.detach() => 不detach试试 ???
        # else:
        # #     # 非 RGB：读取缓存的对应层条件
        #     if rgb_layer_conds is not None and self.block_id is not None and self.block_id >= 0:
        #         rgb_cond_block = rgb_layer_conds['ln_1'][self.block_id]
        #     # 兼容旧接口：若传入了单一 rgb_cond 也可使用
            # if rgb_cond_block is None and rgb_cond is not None:
        #         rgb_cond_block = rgb_cond

        # if input_img_type != InputImageType.rgb and use_cond:
        # #     # 决定当前使用 adaLN 还是全局仿射（训练随机选择；无 rgb 条件或评估时走全局） TODO 加约束损失???
        #     # use_ada = (rgb_cond_block is not None) and self.training and (torch.rand(1, device=ln1_out.device) > self.global_affine_prob)
        # #     # 选择模态 key
        #     if input_img_type == InputImageType.sketch:
        #         key_m = "sketch"
        #     elif input_img_type == InputImageType.color_pencil:
        #         key_m = "color_pencil"
        #     elif input_img_type == InputImageType.nir:
        #         key_m = "nir"
        #     else:
        #         key_m = None

        #     if key_m is not None:
        #         dims = ln1_out.shape  # [L, B, D]
        #         B_axis = 1 if len(dims) >= 2 else 0
        #         B_val = dims[B_axis] if len(dims) >= 2 else 1
        #         if self.training and (rgb_cond_block is not None):
        #             gamma_linear = self.gamma1_proj[key_m](rgb_cond_block)  # [B, D]
        #             beta_linear = self.beta1_proj[key_m](rgb_cond_block)    # [B, D]
        #             gamma_global = self.gamma1_global[key_m].unsqueeze(0).expand(B_val, -1).to(gamma_linear.dtype)
        #             beta_global = self.beta1_global[key_m].unsqueeze(0).expand(B_val, -1).to(beta_linear.dtype)
        #             # reg_ln1 = F.mse_loss(gamma_global, gamma_linear) + F.mse_loss(beta_global, beta_linear)
        #             # 换成相似度loss
        #             # reg_ln1 = 1 - F.cosine_similarity(gamma_global, gamma_linear, dim=1) + 1 - F.cosine_similarity(beta_global, beta_linear, dim=1)
        #             # 统一为缓冲区 dtype（float32），避免 half/float 混合反传错误
        #             # self.adaln_reg = (reg_ln1.mean() * self.adaln_match_loss_weight).to(self.adaln_reg.dtype)
        #             # self.adaln_reg_count = (self.adaln_reg_count + torch.tensor(1.0, device=ln1_out.device)).to(self.adaln_reg_count.dtype)
        #             # 训练的时候两部分加权求和
        #             gamma = gamma_linear + gamma_global
        #             beta = beta_linear + beta_global
        #             # shift_1, scale_1, gate_1, shift_2, scale_2, gate_2 = self.adaLN_modulation_proj[key_m](rgb_cond_block).chunk(6, dim=-1)
        #         else:
        #             # 全局参数展开为 [B, D]
        #             gamma = self.gamma1_global[key_m].unsqueeze(0).expand(B_val, -1)
        #             beta = self.beta1_global[key_m].unsqueeze(0).expand(B_val, -1)
        #             # scale_1 = self.scale_1_global[key_m].unsqueeze(0).expand(B_val, -1).to(ln1_out.dtype)
        #             # shift_1 = self.shift_1_global[key_m].unsqueeze(0).expand(B_val, -1).to(ln1_out.dtype)
        #             # gate_1 = self.gate_1_global[key_m].unsqueeze(0).expand(B_val, -1).to(ln1_out.dtype)
        #             # scale_2 = self.scale_2_global[key_m].unsqueeze(0).expand(B_val, -1).to(ln1_out.dtype)
        #             # shift_2 = self.shift_2_global[key_m].unsqueeze(0).expand(B_val, -1).to(ln1_out.dtype)
        #             # gate_2 = self.gate_2_global[key_m].unsqueeze(0).expand(B_val, -1).to(ln1_out.dtype)

        #         # 自适配广播：根据 batch 轴动态构造 [1, B, D] 或 [B, 1, D]
        #         B_cur = gamma.shape[0]
        #         D_cur = gamma.shape[1]
        #         if len(dims) == 3:
        #             if dims[0] == B_cur:
        #                 shape_mod = (B_cur, 1, D_cur)
        #             elif dims[1] == B_cur:
        #                 shape_mod = (1, B_cur, D_cur)
        #             else:
        #                 shape_mod = (1, B_cur, D_cur)
        #         else:
        #             shape_mod = (1, B_cur, D_cur)
        #         gamma_t = gamma.to(ln1_out.dtype).view(shape_mod)
        #         beta_t = beta.to(ln1_out.dtype).view(shape_mod)
        #         ln1_out = ln1_out * gamma_t + beta_t

                # ln1_out = (1 + scale_1) * ln1_out + shift_1

        #     x = x + gate_1.unsqueeze(0) * self.attention((1 + scale_1.unsqueeze(0)) * ln1_out + shift_1.unsqueeze(0))
        # else:
        #     x = x + self.attention(ln1_out)

        ## MHSA multi head self attention（共享）
        x = x + self.attention(ln1_out)
        
        
        # x = x + outputs_[0]
        residual = x
        # 共享 MLP，使用对应模态的 LN2（可选 adaLN 由 RGB 条件调制）
        ln2_out = ln2(x)
        # rgb_cond_block_2 = None
        # if input_img_type == InputImageType.rgb:
        #     if rgb_layer_conds is not None and self.block_id is not None and self.block_id >= 0:
        # #         # ln1_out: [L, B, D] -> [B, D]
        #         # rgb_cond_block = ln1_out.mean(dim=0)  # TODO 取平均是否合理 ??? 这个是rgb的ln输出, 取rgb经过ln1之后的特征吗??? 应该是经过ln1之前的?? TODO test
        # #         # 只取cls token
        #         rgb_cond_block_2 = x.mean(dim=0)#ln2_out[0, :, :]
        #         # rgb_cond_block = ln1_out
        #         rgb_layer_conds['ln_2'][self.block_id] = rgb_cond_block_2
        # else:
        #     if rgb_layer_conds is not None and self.block_id is not None and self.block_id >= 0:
        #         rgb_cond_block_2 = rgb_layer_conds['ln_2'][self.block_id]

        if input_img_type != InputImageType.rgb and use_cond:
            # 随机选择是否用线性输出的还是可学习的? 推理的时候就用的可学习的啊???那线性映射就没用了啊 TODO 
            # use_ada2 = (rgb_cond_block_2 is not None) and self.training and (torch.rand(1, device=ln2_out.device) > self.global_affine_prob)
            if input_img_type == InputImageType.sketch:
                key_m = "sketch"
            elif input_img_type == InputImageType.color_pencil:
                key_m = "color_pencil"
            elif input_img_type == InputImageType.nir:
                key_m = "nir"
            else:
                key_m = None
            if key_m is not None:
                # dims2 = ln2_out.shape
                # B_axis2 = 1 if len(dims2) >= 2 else 0
                # B_val2 = dims2[B_axis2] if len(dims2) >= 2 else 1
                # if use_ada2:
                #     gamma2 = self.gamma2_proj[key_m](rgb_cond_block_2)
                #     beta2 = self.beta2_proj[key_m](rgb_cond_block_2)
                #     # gamma2_global = self.gamma2_global[key_m].unsqueeze(0).to(gamma2_linear.dtype)
                #     # beta2_global = self.beta2_global[key_m].unsqueeze(0).to(beta2_linear.dtype)
                #     # reg_ln2 = F.mse_loss(gamma2_global, gamma2_linear) + F.mse_loss(beta2_global, beta2_linear)
                #     # self.adaln_reg = (self.adaln_reg + reg_ln2 * self.adaln_match_loss_weight).to(self.adaln_reg.dtype)
                #     # self.adaln_reg_count = (self.adaln_reg_count + torch.tensor(1.0, device=ln2_out.device)).to(self.adaln_reg_count.dtype)
                #     # gamma2 = gamma2_linear + gamma2_global
                #     # beta2 = beta2_linear + beta2_global
                # else:
                #     gamma2 = self.gamma2_global[key_m].unsqueeze(0).expand(B_val2, -1).to(ln2_out.dtype)
                #     beta2 = self.beta2_global[key_m].unsqueeze(0).expand(B_val2, -1).to(ln2_out.dtype)

                # B_cur = gamma2.shape[0]
                # D_cur = gamma2.shape[1]
                # if len(dims2) == 3:
                #     if dims2[0] == B_cur:
                #         shape_mod2 = (B_cur, 1, D_cur)
                #     elif dims2[1] == B_cur:
                #         shape_mod2 = (1, B_cur, D_cur)
                #     else:
                #         shape_mod2 = (1, B_cur, D_cur)
                # else:
                #     shape_mod2 = (1, B_cur, D_cur)
                # gamma2_t = gamma2.to(ln2_out.dtype).view(shape_mod2)
                # beta2_t = beta2.to(ln2_out.dtype).view(shape_mod2)
                # ln2_out = ln2_out * gamma2_t + beta2_t

                # 直接拿rgb的gamma, beta调制
                ln2_out = ln2_out * (self.ln_2.weight.half() + self.gamma2_global[key_m].half()) + \
                             (self.ln_2.bias.half() + self.beta2_global[key_m].half())

        #     original_x = gate_2.unsqueeze(0) * self.mlp((1 + scale_2.unsqueeze(0)) * ln2_out + shift_2.unsqueeze(0))
        # else:
        #     original_x = self.mlp(ln2_out)
        original_x = self.mlp(ln2_out)

        # adapter mlp 分支关闭：仅保留共享 MLP，满足“仅 LN1/LN2 不共享，其余共享”
        x = original_x + residual

        return x, input_img_type, (rgb_layer_conds if rgb_layer_conds is not None else rgb_cond)#, atten_weight


class ResidualAttentionBlockShareAll(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None, args = None):  # args默认是 None, 防止修改其他调用改类的内容
        super().__init__()

        """
        所有共享
        """
        # 注意：仅保留 LN1/LN2 为非共享，其余层（Attention/MLP）全部共享
        # 使用标准 MultiheadAttention 以确保共享
        self.attn = nn.MultiheadAttention(d_model, n_head)
        # 运行时注入的层索引，由构建器设置
        self.block_id = -1
        
        # 为不同模态创建各自独立的 LN1/LN2，其余权重共享
        self.ln_1 = LayerNorm(d_model)
        self.args = args
        # 训练中在 adaLN 与全局仿射之间随机选择的概率（选择全局仿射的概率）
        self.global_affine_prob = getattr(self.args, 'global_affine_prob', 0.5) if self.args is not None else 0.5
        # 仅前 N 层对非 RGB 模态使用来自 RGB 的条件，后续层改为各自 LN 的仿射参数
        self.cond_layers = getattr(self.args, 'cond_layers', 2) if self.args is not None else 2
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))

        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        # 使用共享的标准 MultiheadAttention
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    # def forward(self, x: torch.Tensor, input_img_type: InputImageType = None):
    def forward(self, new_x):
        """注意输入, nn.sequential只能单输入; 得到的x维度是 [n, bs, dim]"""
        assert isinstance(new_x, tuple)
        # 兼容三元组输入：(x, input_img_type, cond_holder)
        # cond_holder: 可以是 Tensor[B, D]（旧逻辑）或 List[Optional[Tensor[B,D]]]（分层条件缓存）
        if len(new_x) >= 3:
            x, input_img_type, cond_holder = new_x[0], new_x[1], new_x[2]
        else:
            x, input_img_type = new_x[0], new_x[1]
            cond_holder = None
        # 解析分层条件缓存
        rgb_layer_conds = cond_holder if isinstance(cond_holder, list) else None
        rgb_cond = cond_holder if (cond_holder is not None and not isinstance(cond_holder, list)) else None
        ln1 = self.ln_1
        ln2 = self.ln_2

        # LN1 -> (可选 adaLN 由 RGB 条件调制) -> Attention（共享）
        ln1_out = ln1(x)
        x = x + self.attention(ln1_out)
        # x = x + outputs_[0]
        residual = x
        # 共享 MLP，使用对应模态的 LN2（可选 adaLN 由 RGB 条件调制）
        ln2_out = ln2(x)
        original_x = self.mlp(ln2_out)

        # adapter mlp 分支关闭：仅保留共享 MLP，满足“仅 LN1/LN2 不共享，其余共享”
        x = original_x + residual

        return x, input_img_type, (rgb_layer_conds if rgb_layer_conds is not None else rgb_cond)#, atten_weight

class Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers)])

    def forward(self, x: torch.Tensor):
        return self.resblocks(x)



class ResidualAttentionBlockTextAdaLN(nn.Module):
    def __init__(self, d_model: int, n_head: int, cond_dim: int, attn_mask: torch.Tensor = None, args=None):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_head)
        # 文本侧 LN 不带可学习参数，由外部 gamma/beta 调制
        # self.ln_1_txt = LayerNorm(d_model, elementwise_affine=False)
        # self.ln_2_txt = LayerNorm(d_model, elementwise_affine=False)
        # 后续层使用的文本侧“自有”仿射 LN
        self.ln_1 = LayerNorm(d_model)
        self.ln_2 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.attn_mask = attn_mask
        self.args = args
        self.global_affine_prob = getattr(self.args, 'global_affine_prob', 0.5) if self.args is not None else 0.5
        # 运行时注入的层索引，由构建器设置
        self.block_id = -1
        # 仅前 N 层使用来自视觉 RGB 的条件
        self.cond_layers = getattr(self.args, 'cond_layers', 2) if self.args is not None else 2

        self.shared_layers = getattr(self.args, 'shared_layers', [0, 6]) if self.args is not None else [0, 6]
        # 将视觉 RGB 条件（cond_dim）映射到文本维度 d_model
        # self.gamma1_proj_txt = nn.Linear(cond_dim, d_model)
        # self.beta1_proj_txt = nn.Linear(cond_dim, d_model)
        # self.gamma2_proj_txt = nn.Linear(cond_dim, d_model)
        # self.beta2_proj_txt = nn.Linear(cond_dim, d_model)

        # self.adapLN_modulation_proj_txt = nn.Sequential(
        #         nn.SiLU(),
        #         nn.Linear(cond_dim, 6 * d_model, bias=True)
        #     )
        # 文本侧全局可学习仿射（每层）
        #
        # self.gamma1_global = nn.Parameter(torch.ones(d_model))
        # self.beta1_global = nn.Parameter(torch.zeros(d_model))
        # self.gamma2_global = nn.Parameter(torch.ones(d_model))
        # self.beta2_global = nn.Parameter(torch.zeros(d_model))

        # self.scale_1_global_txt = nn.Parameter(torch.zeros(d_model))
        # self.shift_1_global_txt = nn.Parameter(torch.zeros(d_model))
        # self.gate_1_global_txt = nn.Parameter(torch.ones(d_model))
        # self.scale_2_global_txt = nn.Parameter(torch.zeros(d_model))
        # self.shift_2_global_txt = nn.Parameter(torch.zeros(d_model))
        # self.gate_2_global_txt = nn.Parameter(torch.ones(d_model))

    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, new_x):
        # 输入为 (x, cond_holder)
        assert isinstance(new_x, tuple)
        if len(new_x) >= 2:
            x, cond_holder = new_x[0], new_x[1]
        else:
            x, cond_holder = new_x[0], None
        # rgb_layer_conds = cond_holder if isinstance(cond_holder, list) else None
        rgb_cond = cond_holder if (cond_holder is not None and not isinstance(cond_holder, list)) else None

        # # 是否在本层使用条件（前 N 层）
        # use_cond = (self.block_id is not None) and (self.block_id < self.cond_layers)
        # LN1
        ln1_mod = self.ln_1
        ln1_out = ln1_mod(x)
        # 取对应层条件
        # rgb_cond_block = None
        # if rgb_layer_conds is not None and self.block_id is not None and self.block_id >= 0:
        #     rgb_cond_block = rgb_layer_conds[self.block_id]
        # if rgb_cond_block is None and rgb_cond is not None:
        #     rgb_cond_block = rgb_cond
        # use_ada_txt = use_cond and (rgb_cond_block is not None) and self.training and (torch.rand(1, device=ln1_out.device) > self.global_affine_prob)
        # if use_cond and ((rgb_cond_block is not None) or (not self.training)):
        #     if use_ada_txt and rgb_cond_block is not None:
        #         # gamma = self.gamma1_proj_txt(rgb_cond_block)  # [B, d_model]
        #         # beta = self.beta1_proj_txt(rgb_cond_block)    # [B, d_model]
        #         shift_1, scale_1, gate_1, shift_2, scale_2, gate_2 = self.adapLN_modulation_proj_txt(rgb_cond_block).chunk(6, dim=-1)
        #     else:
        #         dims = ln1_out.shape
        #         B_axis = 1 if len(dims) >= 2 else 0
        #         B_val = dims[B_axis] if len(dims) >= 2 else 1
        #         # gamma = self.gamma1_global.unsqueeze(0).expand(B_val, -1)  # [B, d_model]
        #         # beta = self.beta1_global.unsqueeze(0).expand(B_val, -1)    # [B, d_model]
        #         shift_1 = self.shift_1_global_txt.unsqueeze(0).expand(B_val, -1).to(ln1_out.dtype)
        #         scale_1 = self.scale_1_global_txt.unsqueeze(0).expand(B_val, -1).to(ln1_out.dtype)
        #         gate_1 = self.gate_1_global_txt.unsqueeze(0).expand(B_val, -1).to(ln1_out.dtype)
        #         shift_2 = self.shift_2_global_txt.unsqueeze(0).expand(B_val, -1).to(ln1_out.dtype)
        #         scale_2 = self.scale_2_global_txt.unsqueeze(0).expand(B_val, -1).to(ln1_out.dtype)
        #         gate_2 = self.gate_2_global_txt.unsqueeze(0).expand(B_val, -1).to(ln1_out.dtype)

        #     # dims = ln1_out.shape
        #     # B_cur = gamma.shape[0]
        #     # D_cur = gamma.shape[1]
        #     # if len(dims) == 3:
        #     #     if dims[0] == B_cur:
        #     #         shape_mod = (B_cur, 1, D_cur)
        #     #     elif dims[1] == B_cur:
        #     #         shape_mod = (1, B_cur, D_cur)
        #     #     else:
        #     #         shape_mod = (1, B_cur, D_cur)
        #     # else:
        #     #     shape_mod = (1, B_cur, D_cur)
        #     # gamma_t = gamma.to(ln1_out.dtype).view(shape_mod)
        #     # beta_t = beta.to(ln1_out.dtype).view(shape_mod)
        #     # ln1_out = ln1_out * gamma_t + beta_t
        x = x + self.attention(ln1_out)

        #     x = x + gate_1.unsqueeze(0) * self.attention((1 + scale_1.unsqueeze(0)) * ln1_out + shift_1.unsqueeze(0))
        # else:
        #     x = x + self.attention(ln1_out)

        # LN2 + adaLN 调制 -> MLP
        residual = x
        # LN2
        ln2_mod = self.ln_2
        ln2_out = ln2_mod(x)
        # use_ada_txt2 = use_cond and (rgb_cond_block is not None) and self.training and (torch.rand(1, device=ln2_out.device) > self.global_affine_prob)
        # if use_cond and ((rgb_cond_block is not None) or (not self.training)):
        #     # if use_ada_txt2 and rgb_cond_block is not None:
        # #         gamma2 = self.gamma2_proj_txt(rgb_cond_block)
        # #         beta2 = self.beta2_proj_txt(rgb_cond_block)
                
        #     # else:
        # #         dims2 = ln2_out.shape
        # #         B_axis2 = 1 if len(dims2) >= 2 else 0
        # #         B_val2 = dims2[B_axis2] if len(dims2) >= 2 else 1
        # #         gamma2 = self.gamma2_global.unsqueeze(0).expand(B_val2, -1)
        # #         beta2 = self.beta2_global.unsqueeze(0).expand(B_val2, -1)
        # #     dims2 = ln2_out.shape
        # #     B_cur = gamma2.shape[0]
        # #     D_cur = gamma2.shape[1]
        # #     if len(dims2) == 3:
        # #         if dims2[0] == B_cur:
        # #             shape_mod2 = (B_cur, 1, D_cur)
        # #         elif dims2[1] == B_cur:
        # #             shape_mod2 = (1, B_cur, D_cur)
        # #         else:
        # #             shape_mod2 = (1, B_cur, D_cur)
        # #     else:
        # #         shape_mod2 = (1, B_cur, D_cur)
        # #     gamma2_t = gamma2.to(ln2_out.dtype).view(shape_mod2)
        # #     beta2_t = beta2.to(ln2_out.dtype).view(shape_mod2)
        # #     ln2_out = ln2_out * gamma2_t + beta2_t
        x = self.mlp(ln2_out) + residual

        #     x = gate_2.unsqueeze(0) * self.mlp((1 + scale_2.unsqueeze(0)) * ln2_out + shift_2.unsqueeze(0)) + residual
        # else:
        #     x = self.mlp(ln2_out) + residual

        return x, rgb_cond


class ResidualAttentionBlockTextShareAll(nn.Module):
    def __init__(self, d_model: int, n_head: int, cond_dim: int, attn_mask: torch.Tensor = None, args=None):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.ln_2 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.attn_mask = attn_mask
        self.args = args
        self.global_affine_prob = getattr(self.args, 'global_affine_prob', 0.5) if self.args is not None else 0.5
        # 运行时注入的层索引，由构建器设置
        self.block_id = -1
        # 仅前 N 层使用来自视觉 RGB 的条件
        self.cond_layers = getattr(self.args, 'cond_layers', 2) if self.args is not None else 2

    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, new_x):
        # 输入为 (x, cond_holder)
        assert isinstance(new_x, tuple)
        if len(new_x) >= 2:
            x, cond_holder = new_x[0], new_x[1]
        else:
            x, cond_holder = new_x[0], None
        # rgb_layer_conds = cond_holder if isinstance(cond_holder, list) else None
        rgb_cond = cond_holder if (cond_holder is not None and not isinstance(cond_holder, list)) else None
        ln1_mod = self.ln_1
        ln1_out = ln1_mod(x)
        x = x + self.attention(ln1_out)
        residual = x
        # LN2
        ln2_mod = self.ln_2
        ln2_out = ln2_mod(x)
        x = self.mlp(ln2_out) + residual

        return x, rgb_cond

class TransformerTextAdaLN(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, cond_dim: int, attn_mask: torch.Tensor = None, args=None):
        super().__init__()
        self.width = width
        self.layers = layers
        blocks = [
            ResidualAttentionBlockTextAdaLN(width, heads, cond_dim, attn_mask, args=args) for _ in range(layers)
        ]
        for idx, b in enumerate(blocks):
            b.block_id = idx
        self.resblocks = nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor, cond_holder=None):
        new_x = (x, cond_holder)
        return self.resblocks(new_x)


############# 最原始的transformer ########################
class ResidualAttentionBlockOrignal(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        # 0722: need_weight改为True, 返回attention maps
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x):
        # ln_1 -> attention -> + x -> ln_2 -> mlp -> + x(这个x是第一次加完后的 !!)
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class TransformerOrignal(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(*[ResidualAttentionBlockOrignal(width, heads, attn_mask) for _ in range(layers)])
    def forward(self, x: torch.Tensor):
        return self.resblocks(x)

# 暂时舍弃这种组合方式
# 自定义一个模块来组合这些模块
# 注意: 此方式行不通, 会噪声损失偏大, 原因可能是没有将其加入到优化器adam里面
# class CustomSequential(nn.Module):
#     def __init__(self, modules):
#         super(CustomSequential, self).__init__()
#         self.module_list = nn.ModuleList(modules)  # 使用 ModuleList 来存储模块

#     def forward(self, x, y):
#         for module in self.module_list:
#             x, y = module(x, y)  # 每个模块的 forward 方法需要两个参数
#         return x

class IBN(nn.Module):
    def __init__(self, planes):
        super(IBN, self).__init__()
        half1 = int(planes/2)
        self.half1 = half1
        half2 = planes - half1
        self.IN = nn.InstanceNorm2d(half1, affine=True)
        self.BN = nn.BatchNorm2d(half2)  ### affine默认是 True

        nn.init.constant_(self.IN.weight, 1.0)
        nn.init.constant_(self.IN.bias, 0.0)
        nn.init.constant_(self.BN.weight, 1.0)
        nn.init.constant_(self.BN.bias, 0.0)

    def forward(self, x):
        split = torch.split(x, self.half1, 1)
        out1 = self.IN(split[0].contiguous())
        out2 = self.BN(split[1].contiguous())
        out = torch.cat((out1, out2), 1)
        return out


class TransformerWithAdapter(nn.Module):
    """构建包含adapterformer的transformer, 把vit的mlp替换"""
    def __init__(self, 
                 args, 
                 width: int, 
                 layers: int, 
                 heads: int, 
                 attn_mask: torch.Tensor = None):
        super().__init__()
        self.width = width
        self.layers = layers
        blocks = [ResidualAttentionBlockWithAdapter(width, heads, attn_mask, args=args) for _ in range(layers)]
        for idx, b in enumerate(blocks):
            b.block_id = idx
        self.resblocks = nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor, input_img_type: InputImageType = None, cond_holder=None):#, attn_weight = None):
        """注意: sequential只能传入单个参数, forward不支持传入多个, 需要改写"""
        new_x = (x, input_img_type, cond_holder)
        return self.resblocks(new_x)


from timm.models.vision_transformer import PatchEmbed, Attention, Mlp


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class DiTBlock(nn.Module):
    """
    A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning.
    """
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, **block_kwargs):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
        x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x



class SwitchNorm2d(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.9, using_moving_average=True, using_bn=True, last_gamma=False):
        super(SwitchNorm2d, self).__init__()
        self.eps = eps
        self.momentum = momentum
        self.using_moving_average = using_moving_average
        self.using_bn = using_bn
        self.last_gamma = last_gamma
        self.weight = nn.Parameter(torch.ones(1, num_features, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, num_features, 1, 1))
        if self.using_bn:
            self.mean_weight = nn.Parameter(torch.ones(3))
            self.var_weight = nn.Parameter(torch.ones(3))
        else:
            self.mean_weight = nn.Parameter(torch.ones(2))
            self.var_weight = nn.Parameter(torch.ones(2))
        if self.using_bn:
            self.register_buffer('running_mean', torch.zeros(1, num_features, 1))
            self.register_buffer('running_var', torch.zeros(1, num_features, 1))

        self.reset_parameters()

    def reset_parameters(self):
        if self.using_bn:
            self.running_mean.zero_()
            self.running_var.zero_()
        if self.last_gamma:
            self.weight.data.fill_(0)
        else:
            self.weight.data.fill_(1)
        self.bias.data.zero_()

    def _check_input_dim(self, input):
        if input.dim() != 4:
            raise ValueError('expected 4D input (got {}D input)'
                             .format(input.dim()))

    def forward(self, x):
        self._check_input_dim(x)
        N, C, H, W = x.size()
        x = x.view(N, C, -1)
        mean_in = x.mean(-1, keepdim=True)
        var_in = x.var(-1, keepdim=True)

        mean_ln = mean_in.mean(1, keepdim=True)
        temp = var_in + mean_in ** 2
        var_ln = temp.mean(1, keepdim=True) - mean_ln ** 2

        if self.using_bn:
            if self.training:
                mean_bn = mean_in.mean(0, keepdim=True)
                var_bn = temp.mean(0, keepdim=True) - mean_bn ** 2
                if self.using_moving_average:
                    self.running_mean.mul_(self.momentum)
                    self.running_mean.add_((1 - self.momentum) * mean_bn.data)
                    self.running_var.mul_(self.momentum)
                    self.running_var.add_((1 - self.momentum) * var_bn.data)
                else:
                    self.running_mean.add_(mean_bn.data)
                    self.running_var.add_(mean_bn.data ** 2 + var_bn.data)
            else:
                mean_bn = torch.autograd.Variable(self.running_mean)
                var_bn = torch.autograd.Variable(self.running_var)

        softmax = nn.Softmax(0)
        mean_weight = softmax(self.mean_weight)
        var_weight = softmax(self.var_weight)

        if self.using_bn:
            mean = mean_weight[0] * mean_in + mean_weight[1] * mean_ln + mean_weight[2] * mean_bn
            var = var_weight[0] * var_in + var_weight[1] * var_ln + var_weight[2] * var_bn
        else:
            mean = mean_weight[0] * mean_in + mean_weight[1] * mean_ln
            var = var_weight[0] * var_in + var_weight[1] * var_ln

        x = (x-mean) / (var+self.eps).sqrt()
        x = x.view(N, C, H, W)
        return x * self.weight + self.bias
    


class VisionTransformer(nn.Module):
    def __init__(self, 
                 args,
                 input_resolution: Tuple[int, int], 
                 patch_size: int, 
                 stride_size: int, 
                 width: int, 
                 layers: int, 
                 heads: int, 
                 output_dim: int):
        super().__init__()
        self.input_resolution = input_resolution # (384, 128)
        self.num_x = (input_resolution[1] - patch_size) // stride_size + 1
        self.num_y = (input_resolution[0] - patch_size) // stride_size + 1
        num_patches = self.num_x * self.num_y

        self.output_dim = output_dim
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=width, kernel_size=patch_size, stride=stride_size, bias=False)

        # tokenizer不共享
        self.conv1_si = nn.Conv2d(in_channels=3, out_channels=width, kernel_size=patch_size, stride=stride_size, bias=False)
        self.conv1_ci = nn.Conv2d(in_channels=3, out_channels=width, kernel_size=patch_size, stride=stride_size, bias=False)
        self.conv1_ni = nn.Conv2d(in_channels=3, out_channels=width, kernel_size=patch_size, stride=stride_size, bias=False)
        ### 加入IBN
        # self.ibn = IBN(width)
        # self.ibn_si = IBN(width)
        # self.ibn_ci = IBN(width)
        # self.ibn_ni = IBN(width)
        # self.sn = SwitchNorm2d(width).half()
        # self.sn_si = SwitchNorm2d(width).half()
        # self.sn_ci = SwitchNorm2d(width).half()
        # self.sn_ni = SwitchNorm2d(width).half()

        scale = width ** -0.5 # 1/sqrt(768)
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(scale * torch.randn(num_patches + 1, width))
        self.ln_pre = LayerNorm(width)

        # 设置 sketch, color pencil, nir 3 个 图像的tokenizer
        self.ln_pre_si = LayerNorm(width)
        self.ln_pre_ci = LayerNorm(width)
        self.ln_pre_ni = LayerNorm(width)
        self.class_embedding_si = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding_si = nn.Parameter(scale * torch.randn(num_patches + 1, width))
        self.class_embedding_ci = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding_ci = nn.Parameter(scale * torch.randn(num_patches + 1, width))
        self.class_embedding_ni = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding_ni = nn.Parameter(scale * torch.randn(num_patches + 1, width))

        self.transformer = Transformer(width, layers, heads)
        # self.transformer = TransformerWithAdapter(args, width, layers, heads)  ####

        self.ln_post = LayerNorm(width)
        self.proj = nn.Parameter(scale * torch.randn(width, output_dim))

        self.args = args


    def _pre_forward(self, x, conv1, class_embedding, positional_embedding, ln_pre):#:, ibn):
        x = conv1(x)  # shape = [*, width, grid, grid]
        # x = ibn(x)

        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat(
            [class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], 
            dim=1
            )  # shape = [*, grid ** 2 + 1, width]
        x = x + positional_embedding.to(x.dtype)
        x = ln_pre(x)
        return x


    def forward(self, x: torch.Tensor, input_img_type: InputImageType = None):
        """
        修改点1: conv, cls embed, pos embed, ln 不共享
        修改点2: visual transformer的 ffn 用lora替换/adapter替换, 结合MoE的思路做
        """
        # rgb_cond = None
        if input_img_type == InputImageType.rgb:
            x = self._pre_forward(x, self.conv1, self.class_embedding, self.positional_embedding, self.ln_pre)#, self.sn)
            # self.conv1.requires_grad_(True)
            # self.ln_pre.requires_grad_(True)
            # # self.sn.requires_grad_(True)
            # # self.sn_ci.requires_grad_(False)
            # # self.sn_si.requires_grad_(False)
            # # self.sn_ni.requires_grad_(False)
            # self.conv1_ci.requires_grad_(False)
            # self.conv1_si.requires_grad_(False)
            # self.conv1_ni.requires_grad_(False)
            # self.ln_pre_ci.requires_grad_(False)
            # self.ln_pre_si.requires_grad_(False)
            # self.ln_pre_ni.requires_grad_(False)

            # 从 RGB 预处理后的 token 提取全局条件（B, D），供后续模态的 adaLN 使用
            # rgb_cond = x.mean(dim=1)  # [B, D]
            # 缓存 RGB 条件供文本侧使用
            # self.last_rgb_cond = rgb_cond
            # 初始化/重置分层 RGB 条件缓存（仅在训练时对其他模态可用）
            # if self.training:
            #     self.rgb_layer_conds_cache = {
            #         'ln_1' : [None for _ in range(self.transformer.layers)],
            #         'ln_2' : [None for _ in range(self.transformer.layers)],
            #     }

        elif input_img_type == InputImageType.sketch:
            x = self._pre_forward(x, self.conv1_si, self.class_embedding_si, self.positional_embedding_si, self.ln_pre_si)#, self.sn_si)
            # self.conv1_si.requires_grad_(True)
            # self.ln_pre_si.requires_grad_(True)
            # # self.sn_si.requires_grad_(True)
            # # self.sn_ci.requires_grad_(False)
            # # self.sn.requires_grad_(False)
            # # self.sn_ni.requires_grad_(False)
            # self.conv1_ci.requires_grad_(False)
            # self.conv1.requires_grad_(False)
            # self.conv1_ni.requires_grad_(False)
            # self.ln_pre_ci.requires_grad_(False)
            # self.ln_pre.requires_grad_(False)
            # self.ln_pre_ni.requires_grad_(False)

        elif input_img_type == InputImageType.color_pencil:
            x = self._pre_forward(x, self.conv1_ci, self.class_embedding_ci, self.positional_embedding_ci, self.ln_pre_ci)#, self.sn_ci)
            # self.conv1_ci.requires_grad_(True)
            # self.ln_pre_ci.requires_grad_(True)
            # # self.sn_ci.requires_grad_(True)
            # # self.sn.requires_grad_(False)
            # # self.sn_si.requires_grad_(False)
            # # self.sn_ni.requires_grad_(False)
            # self.conv1.requires_grad_(False)
            # self.conv1_si.requires_grad_(False)
            # self.conv1_ni.requires_grad_(False)
            # self.ln_pre.requires_grad_(False)
            # self.ln_pre_si.requires_grad_(False)
            # self.ln_pre_ni.requires_grad_(False)

        elif input_img_type == InputImageType.nir:
            x = self._pre_forward(x, self.conv1_ni, self.class_embedding_ni, self.positional_embedding_ni, self.ln_pre_ni)#, self.sn_ni)
            # self.conv1_ni.requires_grad_(True)
            # self.ln_pre_ni.requires_grad_(True)
            # # self.sn_ni.requires_grad_(True)
            # # self.sn_ci.requires_grad_(False)
            # # self.sn_si.requires_grad_(False)
            # # self.sn.requires_grad_(False)
            # self.conv1_ci.requires_grad_(False)
            # self.conv1_si.requires_grad_(False)
            # self.conv1.requires_grad_(False)
            # self.ln_pre_ci.requires_grad_(False)
            # self.ln_pre_si.requires_grad_(False)
            # self.ln_pre.requires_grad_(False)

        # x = self._pre_forward(x, self.conv1, self.class_embedding, self.positional_embedding, self.ln_pre)

        x = x.permute(1, 0, 2)  # NLD -> LND
        # 条件传递：训练阶段 RGB 写入每层条件；非 RGB 读取最近 RGB 缓存；推理阶段非 RGB 使用全局仿射（不传条件）
        # if input_img_type == InputImageType.rgb:
        #     cond_holder = self.rgb_layer_conds_cache if self.training else None
        # else:
        #     cond_holder = self.rgb_layer_conds_cache if (self.training and hasattr(self, "rgb_layer_conds_cache")) else None
        # 进入 Transformer 前将一致性损失清零
        # if hasattr(self.transformer, "resblocks"):
        #     for blk in self.transformer.resblocks:
        #         if hasattr(blk, "adaln_reg"):
        #             # 以 float32 清零，避免 half/float 混合
        #             blk.adaln_reg = blk.adaln_reg.detach() * 0.0
        #         if hasattr(blk, "adaln_reg_count"):
        #             blk.adaln_reg_count = blk.adaln_reg_count.detach() * 0.0
        x = self.transformer(x, input_img_type)
        # 汇总所有层的一致性损失
        # total_adaln_reg = torch.tensor(0.0, device=x.device, dtype=torch.float32)
        # total_adaln_cnt = torch.tensor(0.0, device=x.device, dtype=torch.float32)
        # if hasattr(self.transformer, "resblocks"):
        #     for blk in self.transformer.resblocks:
        #         if hasattr(blk, "adaln_reg"):
        #             total_adaln_reg = total_adaln_reg + blk.adaln_reg.to(total_adaln_reg.dtype)
        #         if hasattr(blk, "adaln_reg_count"):
        #             total_adaln_cnt = total_adaln_cnt + blk.adaln_reg_count.to(total_adaln_cnt.dtype)
        # # 取平均值（避免除零）
        # self.adaln_reg_loss = total_adaln_reg / torch.clamp(total_adaln_cnt, min=1.0)
        x = x.permute(1, 0, 2)  # LND -> NLD

        # x = self.ln_post(x[:, 0, :])
        x = self.ln_post(x)

        if self.proj is not None:
            x = x @ self.proj
    
        return x#, atten_weight  ### 0722: 增加返回最后一层attention map



class CLIP(nn.Module):
    def __init__(self,
                 embed_dim: int,
                 # vision
                 image_resolution: Union[int, Tuple[int, int]],
                 vision_layers: Union[Tuple[int, int, int, int], int],
                 vision_width: int,
                 vision_patch_size: int,
                 stride_size: int,
                 # text
                 context_length: int,
                 vocab_size: int,
                 transformer_width: int,
                 transformer_heads: int,
                 transformer_layers: int,
                
                 args  # input parameters for ViT
                 ):
        super().__init__()

        self.context_length = context_length

        if isinstance(vision_layers, (tuple, list)):
            vision_heads = vision_width * 32 // 64
            self.visual = ModifiedResNet(
                layers=vision_layers,
                output_dim=embed_dim,
                heads=vision_heads,
                input_resolution=image_resolution,
                width=vision_width
            )
        else:
            vision_heads = vision_width // 64
            self.visual = VisionTransformer(
                args,
                input_resolution=image_resolution,
                patch_size=vision_patch_size,
                stride_size=stride_size,
                width=vision_width,
                layers=vision_layers,
                heads=vision_heads,
                output_dim=embed_dim
            )
        #### 这个transformer 是text
        # text 是否也可以用 lora, expert? 
        # 文本 Transformer 改为支持 adaLN（由视觉 RGB 条件调制），注意 cond_dim 使用视觉宽度
        self.transformer = TransformerOrignal(
            width=transformer_width,
            layers=transformer_layers,
            heads=transformer_heads,
            cond_dim=vision_width,
            attn_mask=self.build_attention_mask(),
        )
        # 换成 visual 的 transformer
        # self.transformer = self.visual.transformer

        self.vocab_size = vocab_size
        self.token_embedding = nn.Embedding(vocab_size, transformer_width)
        self.positional_embedding = nn.Parameter(torch.empty(self.context_length, transformer_width))
        self.ln_final = LayerNorm(transformer_width)

        self.text_projection = nn.Parameter(torch.empty(transformer_width, embed_dim))
        # self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        self.initialize_parameters()

    # def get_adaln_match_loss(self, reset: bool = False):
    #     """
    #     返回视觉编码器中“全局仿射与线性映射对齐”的一致性损失。
    #     reset=True 时将其清零（仅清 VisionTransformer 聚合值，不清各层参数）。
    #     """
    #     loss = getattr(self.visual, "adaln_reg_loss", torch.tensor(0.0, device=self.text_projection.device))
    #     if reset and hasattr(self.visual, "adaln_reg_loss"):
    #         if isinstance(self.visual.adaln_reg_loss, torch.Tensor):
    #             self.visual.adaln_reg_loss = self.visual.adaln_reg_loss.detach() * 0.0
    #         else:
    #             self.visual.adaln_reg_loss = 0.0
    #     return loss
        
    def initialize_parameters(self):
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.positional_embedding, std=0.01)

        if isinstance(self.visual, ModifiedResNet):
            if self.visual.attnpool is not None:
                std = self.visual.attnpool.c_proj.in_features ** -0.5
                nn.init.normal_(self.visual.attnpool.q_proj.weight, std=std)
                nn.init.normal_(self.visual.attnpool.k_proj.weight, std=std)
                nn.init.normal_(self.visual.attnpool.v_proj.weight, std=std)
                nn.init.normal_(self.visual.attnpool.c_proj.weight, std=std)

            for resnet_block in [self.visual.layer1, self.visual.layer2, self.visual.layer3, self.visual.layer4]:
                for name, param in resnet_block.named_parameters():
                    if name.endswith("bn3.weight"):
                        nn.init.zeros_(param)

        proj_std = (self.transformer.width ** -0.5) * ((2 * self.transformer.layers) ** -0.5)
        attn_std = self.transformer.width ** -0.5
        fc_std = (2 * self.transformer.width) ** -0.5
        for block in self.transformer.resblocks:
        # for block in self.transformer.resblocks.module_list:
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)


        if self.text_projection is not None:
            nn.init.normal_(self.text_projection, std=self.transformer.width ** -0.5)

    def build_attention_mask(self):
        # lazily create causal attention mask, with full attention between the vision tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.empty(self.context_length, self.context_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)  # zero out the lower diagonal
        return mask

    @property
    def dtype(self):
        return self.visual.conv1.weight.dtype

    def encode_image(self, image, input_img_type=None):
        return self.visual(image.type(self.dtype), input_img_type)

    def encode_text(self, text):
        x = self.token_embedding(text).type(self.dtype)  # [batch_size, n_ctx, d_model]

        x = x + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND

        #### text 的transformer也用相同的? 暂时不行, clip的text encoder和image encoder得到的dim不同
        # 0722: 返回attention map
        # 文本侧支持 adaLN：训练时若视觉缓存了“每层 RGB 条件”则按层调制；否则/推理默认用全局仿射
        cond_holder = None
        if hasattr(self, "visual"):
            if self.training:
                cond_holder = getattr(self.visual, "rgb_layer_conds_cache", None)
                if cond_holder is None:
                    cond_holder = getattr(self.visual, "last_rgb_cond", None)
        x, _ = self.transformer(x, cond_holder)
        # outputs = self.transformer([x])
        # x = outputs[0]
        # atten = outputs[1]
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        # x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection
        x = x @ self.text_projection

        return x#, atten

    def forward(self, image, text, input_img_type=None, combs_select=None):
        """注意: 这个forward只适用于training !!!! 一次输入 4 种 image modality; 改为随机选择"""

        ########## 随机组合一个batch ###########
        # image可能是3种模态的任意组合, rgb一定存在
        # encode_image每次只输入一个模态
        text_features = None
        input_img_type = InputImageType.rgb
        image_features_rgb = self.encode_image(image[0], input_img_type)
        image_features_sk, image_features_cp, image_features_nir = None, None, None
        if combs_select[0]:
            input_img_type = InputImageType.sketch
            image_features_sk = self.encode_image(image[1], input_img_type)
        if combs_select[1]:
            input_img_type = InputImageType.color_pencil
            image_features_cp = self.encode_image(image[2], input_img_type)
        if combs_select[2]:
            input_img_type = InputImageType.nir
            image_features_nir = self.encode_image(image[3], input_img_type)
        if combs_select[3]:
            text_features = self.encode_text(text)

        return image_features_rgb, image_features_sk, image_features_cp, image_features_nir, text_features

        # image_features = self.encode_image(image, input_img_type)
        # text_features = self.encode_text(text)

        # return image_features, text_features
    
    
    def load_param(self, state_dict):
        # state_dict: CLIP原始预训练权重
        # 1. 4个visual模态的tokenizer需要加载一下权重
        # 2. adapter mlp的
        param_dict =  {k: v for k, v in state_dict.items() if k in self.state_dict()}

        if 'model' in param_dict:
            param_dict = param_dict['model']
        if 'state_dict' in param_dict:
            param_dict = param_dict['state_dict']
        for k, v in param_dict.items():
            if k == 'visual.positional_embedding' and v.shape != self.visual.positional_embedding.shape:
                v = resize_pos_embed(v, self.visual.positional_embedding, self.visual.num_y, self.visual.num_x)

            elif k == 'positional_embedding' and v.shape != self.positional_embedding.shape:
                v = resize_text_pos_embed(v, self.context_length)
            try:
                self.state_dict()[k].copy_(v)
            except:
                # 这部分肯定加载正常, 不会输出以下内容
                print(f'===========================ERROR occur in copy {k}, {v.shape}=========================')
                print('shape do not match in k :{}: param_dict{} vs self.state_dict(){}'.format(k, v.shape, self.state_dict()[k].shape))
        
        ## 遍历当前的state_dict, 赋值权重
        for k_cur, _ in self.state_dict().items():

            # if 'mha' in k_cur:
            #     origin_k = k_cur.replace('mha', 'attn')
            #     self.state_dict()[k_cur].copy_(param_dict[origin_k])

            if k_cur == 'visual.positional_embedding_si':
                _v = resize_pos_embed(
                    param_dict['visual.positional_embedding'], 
                    self.visual.positional_embedding_si, 
                    self.visual.num_y, 
                    self.visual.num_x)
                self.state_dict()[k_cur].copy_(_v)
            elif k_cur == 'visual.positional_embedding_ci':
                _v = resize_pos_embed(
                    param_dict['visual.positional_embedding'], 
                    self.visual.positional_embedding_ci, 
                    self.visual.num_y, 
                    self.visual.num_x)
                self.state_dict()[k_cur].copy_(_v)
            elif k_cur == 'visual.positional_embedding_ni':
                _v = resize_pos_embed(
                    param_dict['visual.positional_embedding'], 
                    self.visual.positional_embedding_ni, 
                    self.visual.num_y, 
                    self.visual.num_x)
                self.state_dict()[k_cur].copy_(_v)

            elif k_cur in ['visual.class_embedding_si', 'visual.class_embedding_ci', 'visual.class_embedding_ni']:
                self.state_dict()[k_cur].copy_(state_dict['visual.class_embedding'])

            elif k_cur in ['visual.conv1_si.weight', 'visual.conv1_ci.weight', 'visual.conv1_ni.weight']:
                self.state_dict()[k_cur].copy_(state_dict['visual.conv1.weight'])

            elif k_cur in ['visual.ln_pre_si.weight', 'visual.ln_pre_ci.weight', 'visual.ln_pre_ni.weight']:
                self.state_dict()[k_cur].copy_(state_dict['visual.ln_pre.weight'])

            elif k_cur in ['visual.ln_pre_si.bias', 'visual.ln_pre_ci.bias', 'visual.ln_pre_ni.bias']:
                self.state_dict()[k_cur].copy_(state_dict['visual.ln_pre.bias'])

            # ln 层权重, global权重不从zero开始
            elif 'ln_1_si_affine' in k_cur or 'ln_1_ci_affine' in k_cur or 'ln_1_ni_affine' in k_cur or \
                 'ln_2_si_affine' in k_cur or 'ln_2_ci_affine' in k_cur or 'ln_2_ni_affine' in k_cur:
                key_origin = k_cur.replace('_affine', '')
                # si, ci也去掉
                key_origin = key_origin.replace('_si', '').replace('_ci', '').replace('_ni', '')
                self.state_dict()[k_cur].copy_(state_dict[key_origin])

            elif 'gamma1_global' in k_cur:
                key_origin = k_cur.replace('gamma1_global', '').replace('.color_pencil', '').replace('.sketch', '').replace('.nir', '')
                key_origin += 'ln_1.weight'
                self.state_dict()[k_cur].copy_(state_dict[key_origin])

            elif 'gamma2_global' in k_cur:
                key_origin = k_cur.replace('gamma2_global', '').replace('.color_pencil', '').replace('.sketch', '').replace('.nir', '')
                key_origin += 'ln_2.weight'
                self.state_dict()[k_cur].copy_(state_dict[key_origin])

            elif 'beta1_global' in k_cur:
                key_origin = k_cur.replace('beta1_global', '').replace('.color_pencil', '').replace('.sketch', '').replace('.nir', '')
                key_origin += 'ln_1.bias'
                self.state_dict()[k_cur].copy_(state_dict[key_origin])

            elif 'beta2_global' in k_cur:
                key_origin = k_cur.replace('beta2_global', '').replace('.color_pencil', '').replace('.sketch', '').replace('.nir', '')
                key_origin += 'ln_2.bias'
                self.state_dict()[k_cur].copy_(state_dict[key_origin])
                

            ### adapt mlp 暂时都以 adapter former的方式初始化
        # print('') # check ln的权重是否都加载正常: 没有仿射参数的就不加载, 仅 affine加载, 同时rgb和text原来的也正常加载


def resize_pos_embed(posemb, posemb_new, hight, width):
    # Rescale the grid of position embeddings when loading from state_dict. Adapted from
    # https://github.com/google-research/vision_transformer/blob/00883dd691c63a6830751563748663526e811cee/vit_jax/checkpoint.py#L224
    posemb = posemb.unsqueeze(0)
    posemb_new = posemb_new.unsqueeze(0)

    posemb_token, posemb_grid = posemb[:, :1], posemb[0, 1:]

    gs_old = int(math.sqrt(len(posemb_grid)))
    print('Resized position embedding from size:{} to size: {} with height:{} width: {}'.format(posemb.shape, posemb_new.shape, hight, width))
    posemb_grid = posemb_grid.reshape(1, gs_old, gs_old, -1).permute(0, 3, 1, 2)
    posemb_grid = F.interpolate(posemb_grid, size=(hight, width), mode='bilinear')
    posemb_grid = posemb_grid.permute(0, 2, 3, 1).reshape(1, hight * width, -1)
    posemb = torch.cat([posemb_token, posemb_grid], dim=1)
    return posemb.squeeze(0)


def resize_text_pos_embed(posemb, length):
    old_h, old_w = posemb.shape
    print(f'Resized position embedding from size:{old_h} * {old_w} to size: {length} * {old_w}')

    posemb = posemb.reshape(1, 1, old_h, old_w)  # [1, 1, 77, 512]
    posemb = F.interpolate(posemb, length, mode='bilinear')

    return posemb.squeeze(0)


def convert_weights(model: nn.Module):
    """Convert applicable model parameters to fp16"""

    def _convert_weights_to_fp16(l):
        if isinstance(l, (nn.Conv1d, nn.Conv2d, nn.Linear)):
            l.weight.data = l.weight.data.half()
            if l.bias is not None:
                l.bias.data = l.bias.data.half()

        if isinstance(l, nn.MultiheadAttention):
            for attr in [*[f"{s}_proj_weight" for s in ["in", "q", "k", "v"]], "in_proj_bias", "bias_k", "bias_v"]:
                tensor = getattr(l, attr)
                if tensor is not None:
                    tensor.data = tensor.data.half()

        for name in ["text_projection", "proj", "mcq_proj"]:
            if hasattr(l, name):
                attr = getattr(l, name)
                if attr is not None:
                    attr.data = attr.data.half()

    model.apply(_convert_weights_to_fp16)


def build_CLIP_from_openai_pretrained(
        args, 
        name: str, 
        image_size: Union[int, Tuple[int, int]], 
        stride_size: int, 
        jit: bool = False, 
        download_root: str = None):
    """Load a CLIP model

    Parameters
    ----------
    args: setting parameters

    name : str
        A model name listed by `clip.available_models()`, or the path to a model checkpoint containing the state_dict
    
    image_size: Union[int, Tuple[int, int]]
        Input image size, in Re-ID task, image size commonly set to 384x128, instead of 224x224

    jit : bool
        Whether to load the optimized JIT model or more hackable non-JIT model (default).

    download_root: str
        path to download the model files; by default, it uses "~/.cache/clip"

    Returns
    -------
    model : torch.nn.Module
        The CLIP model
    """
    if name in _MODELS:
        model_path = _download(_MODELS[name], download_root or os.path.expanduser("~/.cache/clip"))
    elif os.path.isfile(name):
        model_path = name
    else:
        raise RuntimeError(f"Model {name} not found; available models = {available_models()}")

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu")
        state_dict = None
    except RuntimeError:
        # loading saved state dict
        if jit:
            warnings.warn(f"File {model_path} is not a JIT archive. Loading as a state dict instead")
            jit = False
        state_dict = torch.load(model_path, map_location="cpu")

    state_dict = state_dict or model.state_dict()

    vit = "visual.proj" in state_dict

    if vit:
        vision_width = state_dict["visual.conv1.weight"].shape[0]
        vision_layers = len([k for k in state_dict.keys() if k.startswith("visual.") and k.endswith(".attn.in_proj_weight")])
        vision_patch_size = state_dict["visual.conv1.weight"].shape[-1]
        grid_size = round((state_dict["visual.positional_embedding"].shape[0] - 1) ** 0.5)
        image_resolution = vision_patch_size * grid_size
    else:
        counts: list = [len(set(k.split(".")[2] for k in state_dict if k.startswith(f"visual.layer{b}"))) for b in [1, 2, 3, 4]]
        vision_layers = tuple(counts)
        vision_width = state_dict["visual.layer1.0.conv1.weight"].shape[0]
        output_width = round((state_dict["visual.attnpool.positional_embedding"].shape[0] - 1) ** 0.5)
        vision_patch_size = None
        assert output_width ** 2 + 1 == state_dict["visual.attnpool.positional_embedding"].shape[0]
        image_resolution = output_width * 32

    embed_dim = state_dict["text_projection"].shape[1]
    context_length = state_dict["positional_embedding"].shape[0]
    vocab_size = state_dict["token_embedding.weight"].shape[0]
    transformer_width = state_dict["ln_final.weight"].shape[0]
    transformer_heads = transformer_width // 64
    transformer_layers = len(set(k.split(".")[2] for k in state_dict if k.startswith(f"transformer.resblocks")))

    model_cfg = {
        'embed_dim': embed_dim,
        'image_resolution': image_resolution,
        'vision_layers': vision_layers, 
        'vision_width': vision_width, 
        'vision_patch_size': vision_patch_size,
        'context_length': context_length, 
        'vocab_size': vocab_size, 
        'transformer_width': transformer_width, 
        'transformer_heads': transformer_heads, 
        'transformer_layers': transformer_layers,

        'args': args
    }


    # modify image resolution to adapt Re-ID task
    model_cfg['image_resolution'] = image_size
    model_cfg['stride_size'] = stride_size
    logger.info(f"Load pretrained {name} CLIP model with model config: {model_cfg}")
    model = CLIP(**model_cfg)

    # covert model to fp16
    # convert_weights(model)

    # resize modified pos embedding
    model.load_param(state_dict)
    return model, model_cfg


### SigLIP 输入必须是 方形的, 不好改为矩形....
# if __name__ == "__main__":
#     from transformers import AutoTokenizer, AutoModel
#     import torch

#     # 1. 如果已离线，可跳过 snapshot_download，直接指向本地路径
#     model_dir = "/home/dj_2025/competitions/UNIReID-main/siglip_naflex"

#     # 2. 加载 tokenizer（Gemma）
#     # tokenizer = AutoTokenizer.from_pretrained(model_dir)
#     # model = AutoModel.from_pretrained(model_dir, device_map="cpu")

#     from transformers import Siglip2Model, SiglipConfig, AutoProcessor
#     # config = SiglipConfig.from_pretrained(model_dir)
#     # # 或者手动覆盖：
#     # config.vision_config.image_size = (384, 128)

#     model = Siglip2Model.from_pretrained(
#         model_dir,
#         # config=config,
#         torch_dtype=torch.float16,
#         device_map="cuda:0",
#     ).eval()

#     processor = AutoProcessor.from_pretrained(model_dir)

#     # load the image
#     image = torch.randn(3, 384, 128).to("cuda:0")
#     inputs = processor(images=[image], return_tensors="pt").to(model.device)

#     txt = ["A lean young man of medium height and jet-black short hair. He wears a maroon long-sleeved jacket on top, \
#         and a pair of black trousers that reach down to his ankles. On his feet are a pair of gray sneakers."]
#     tokenizer = AutoTokenizer.from_pretrained(model_dir)

#     inputs_txt = tokenizer(txt, return_tensors="pt").to(model.device)
#     # run infernece
#     with torch.no_grad():
#         image_embeddings = model.get_image_features(**inputs)    
#         txt_cls = model.get_text_features(inputs_txt["input_ids"]) 

#     print(image_embeddings.shape)

#     clip_model = torch.load("/home/dj_2025/.cache/clip/ViT-B-16.pt", map_location="cpu")
#     # 3. 把权重读进自定义模型
#     # state_dict = torch.load(f"{model_dir}/pytorch_model.bin", map_location="cpu")
#     state_dict = model.state_dict()


