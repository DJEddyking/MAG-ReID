import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import math
import numpy as np

class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb

def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class FinalLayer(nn.Module):
    """
    The final layer of DiT.
    """
    def __init__(self, hidden_size, patch_size):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


class DiTBlock1D(nn.Module):
    """
    DiT Block for 1D data.
    """
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False)
        self.attn = nn.MultiheadAttention(hidden_size, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False)
        
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden_dim),
            nn.GELU(),
            nn.Linear(mlp_hidden_dim, hidden_size),
        )

        # AdaLN-Zero: We'll initialize the scale and shift parameters to zero
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size)  # Predicts scale and shift for norm1, norm2, and the MLP
        )

    def forward(self, x, c):
        # c is the condition embedding (time + text)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
        
        # Modulation for MSA
        x = x + gate_msa.unsqueeze(1) * self.attn(
            self.modulate(self.norm1(x), shift_msa, scale_msa),
            self.modulate(self.norm1(x), shift_msa, scale_msa),
            self.modulate(self.norm1(x), shift_msa, scale_msa),
            need_weights=False
        )[0]
        # Modulation for MLP
        x = x + gate_mlp.unsqueeze(1) * self.mlp(self.modulate(self.norm2(x), shift_mlp, scale_mlp))
        
        return x

    def modulate(self, x, shift, scale):
        return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)



def get_1d_sincos_pos_embed(embed_dim, num_patches):
    """
    embed_dim: 输出位置的维度，应该为偶数
    num_patches: 块的数量
    """
    assert embed_dim % 2 == 0
    # 生成位置序列
    position = torch.arange(0, num_patches, dtype=torch.float32).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, embed_dim, 2, dtype=torch.float32) * (-math.log(10000.0) / embed_dim))
    pos_embed = torch.zeros(num_patches, embed_dim)
    pos_embed[:, 0::2] = torch.sin(position * div_term)
    pos_embed[:, 1::2] = torch.cos(position * div_term)
    return pos_embed



class DiT1D(nn.Module):
    """
    Diffusion Transformer for 1D data.
    """
    def __init__(self, input_size=512, patch_size=16, hidden_size=768, depth=12, num_heads=12, mlp_ratio=4.0, channels=1, self_condition=False):
        super().__init__()
        # self.patch_size = patch_size
        # self.num_patches = input_size // patch_size
        self.hidden_size = hidden_size
        self.channels=channels
        self.self_condition=self_condition

        # Patch embedding: project flattened patches to hidden dimension
        # self.patch_embed = nn.Linear(patch_size, hidden_size)

        # Positional embedding
        # self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, hidden_size), requires_grad=False)#nn.Parameter(torch.zeros(1, self.num_patches, hidden_size), requires_grad=False)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            DiTBlock1D(hidden_size, num_heads, mlp_ratio) for _ in range(depth)
        ])

        # Output layer: predict both noise and covariance (if needed)
        # self.output_layer = nn.Linear(hidden_size, patch_size * 2)  # x2 for mean and variance
        # self.output_layer = FinalLayer(hidden_size, patch_size)

        # Initialize condition embedding for time and text
        self.time_embed = TimestepEmbedder(hidden_size)

        self.text_linear = nn.Linear(512, self.hidden_size)

        self.initialize_weights()

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Initialize (and freeze) pos_embed by sin-cos embedding:
        # pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.num_patches ** 0.5))
        # self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))
        # 一维位置编码
        # pos_embed = get_1d_sincos_pos_embed(self.hidden_size, self.num_patches)
        # self.pos_embed.data.copy_(pos_embed.float().unsqueeze(0))

        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
        # w = self.patch_embed.weight.data
        # nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        # nn.init.constant_(self.patch_embed.bias, 0)

        # Initialize label embedding table:
        # nn.init.normal_(self.y_embedder.embedding_table.weight, std=0.02)

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.time_embed.mlp[0].weight, std=0.02)
        nn.init.normal_(self.time_embed.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers in DiT blocks:
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0.0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        # nn.init.constant_(self.output_layer.adaLN_modulation[-1].weight, 0.0)
        # nn.init.constant_(self.output_layer.adaLN_modulation[-1].bias, 0)
        # nn.init.constant_(self.output_layer.linear.weight, 0.0)
        # nn.init.constant_(self.output_layer.linear.bias, 0)


    def forward(self, x, t, text_emb=None, cond=None):
        """
        x: (B, 1, L) input noisy latent, 只包含了 cls token, 尝试加入局部token???
        t: (B,) timesteps
        text_emb: (B, text_emb_dim) text embeddings from CLIP
        """
        # x = x.squeeze(1)
        # 1. Patchify
        # x = x.unfold(1, self.patch_size, self.patch_size)  # (B, num_patches, patch_size), 假设输入是 bs, 512 => bs, 32, 16
        # x = self.patch_embed(x)  # (B, num_patches, hidden_size) => bs, 32, 768, 将一维序列为patch TODO 其实不对, 应该把seq len设置为 1
        
        # 2. Add positional embedding
        # x = x + self.pos_embed

        # Time embedding can be added similarly. Here we simplify by adding time and text.
        t = self.time_embed(t)  # (B, hidden_size)
        
        # 3. Prepare condition (time + text)
        # Assuming text_emb is (B, text_emb_dim). You might need to project it to hidden_size first.
        # 加入condition
        if text_emb is not None:
            if text_emb.size(1) != self.hidden_size:
                text_emb = self.text_linear(text_emb)
            c = t + text_emb
        else:
            c = t
        
        # 4. Transformer blocks
        for block in self.blocks:
            x = block(x, c)
        
        # 5. Output
        # x = self.output_layer(x, c)  # (B, num_patches, patch_size)
        # x = x.view(x.size(0), -1)  # (B, L)
        return x, None  # bs, 1, dim
