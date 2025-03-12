from einops import rearrange
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


"""
Adopted from https://github.com/marcdcfischer/PUNet/blob/main/src/modules/blocks/attention.py.
"""
class WindowAttention(nn.Module):
    def __init__(self,
                 dim: int,
                 num_heads: int,
                 attn_drop: float = 0.0,
                 proj_drop: float = 0.0,
                 ):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(
                'WindowAttention: The dimension is not compatible '
                'with the number of heads!')
        head_dim = dim // num_heads
        self.num_heads = num_heads
        self.scale = head_dim ** -0.5

        self.to_q = nn.Linear(dim, dim, bias=False)
        self.to_k = nn.Linear(dim, dim, bias=False)
        self.to_v = nn.Linear(dim, dim, bias=False)
        self.attn_drop = nn.Dropout(attn_drop)

        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self,
                q: torch.Tensor,
                k: torch.Tensor,
                v: torch.Tensor,
                pos_bias: Optional[torch.Tensor] = None,
                mask: Optional[torch.Tensor] = None,
                ):
        q = self.to_q(q)
        k = self.to_k(k)
        v = self.to_v(v)
        q = rearrange(q, 'b p n (h d) -> b p h n d', h=self.num_heads)
        k = rearrange(k, 'b p n (h d) -> b p h n d', h=self.num_heads)
        v = rearrange(v, 'b p n (h d) -> b p h n d', h=self.num_heads)

        attn = torch.einsum(
            'b p h i d, b p h j d -> b p h i j', q, k
        ) * self.scale
        if pos_bias is not None:
            attn += pos_bias
        if mask is not None:
            attn *= mask
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)
        attn = torch.einsum('b p h i j, b p h j d -> b p h i d', attn, v)
        attn = rearrange(attn, 'b p h n d -> b p n (h d)', h=self.num_heads)
        attn = self.proj_drop(self.proj(attn))
        return attn