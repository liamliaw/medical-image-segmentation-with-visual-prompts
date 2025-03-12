from einops import rearrange
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from typing import Sequence, Optional
from .down import PatchMerging
from ..multi_head_attention import WindowAttention, RelativePE

"""
Swin based on https://github.com/marcdcfischer/PUNet/blob/main/src/modules/blocks/attention.py,
Prompting based on https://github.com/marcdcfischer/PUNet/blob/main/src/modules/blocks/sia_block_deep.py.
"""

class ConsecutiveSwinBlocks(nn.Module):
    def __init__(self,
                 hidden_channels: int,
                 num_heads: int,
                 pos_bias_embed_dim: int,
                 max_prompts: int,
                 tokens_per_prompt: int,
                 window_size: Sequence[int],
                 use_token_params: bool = True,
                 shift_size: Sequence[int] = None,
                 down: bool = True,
                 merge_last_dim: bool = True,
                 use_checkpoint: bool = False,
                 out_channels: int = None,
                 proj_drop: float = 0.0,
                 attn_drop: float = 0.0,
                 ):

        super().__init__()
        self.window_size = window_size
        self.shift_size = shift_size if shift_size is not None else \
            tuple(size // 2 for size in window_size)
        self.no_shift = tuple(0 for _ in window_size)
        self.down = down
        self.use_checkpoint = use_checkpoint
        # Two Swin blocks.
        self.swin_blocks = nn.ModuleList([
                SwinTransformerBlock(
                    hidden_channels=hidden_channels,
                    num_heads=num_heads,
                    pos_bias_embed_dim=pos_bias_embed_dim,
                    max_prompts=max_prompts,
                    tokens_per_prompt=tokens_per_prompt,
                    use_token_params=use_token_params,
                    window_size=self.window_size,
                    shift_size=self.no_shift if i == 0 else self.shift_size,
                    attn_drop=attn_drop,
                    proj_drop=proj_drop,
                    use_checkpoint=use_checkpoint,
                ) for i in range(2)
        ])
        if self.down:
            out_channels = 2 * hidden_channels if out_channels is None \
                else out_channels
            self.merge = PatchMerging(
                in_channels=hidden_channels,
                out_channels=out_channels,
                merge_last_dim=merge_last_dim,
            )

    def forward(self, x, p=(None, None)):
        for i, blk in enumerate(self.swin_blocks):
            x = blk(x, p[i])
        if self.down:
            x = self.merge(x)
        return x

    # Backbone parameters.
    def named_parameters_body(self):
        params = []
        for i in range(2):
            params.extend(
                self.swin_blocks[i].named_parameters_body())
        if self.down:
            params.extend(self.merge.named_parameters())
        return params

    # Content bias parameters.
    def named_parameters_bias_content(self):
        params = []
        for i in range(2):
            params.extend(self.swin_blocks[i].named_parameters_bias_content())
        return params

    # Prompt bias parameters.
    def named_parameters_bias_prompt_tokens(self):
        params = []
        for i in range(2):
            params.extend(self.swin_blocks[i].named_parameters_bias_prompt_tokens())
        return params


class SwinTransformerBlock(nn.Module):
    def __init__(self,
                 hidden_channels: int,
                 window_size: Sequence[int],
                 pos_bias_embed_dim: int,
                 num_heads: int,
                 max_prompts: int,
                 tokens_per_prompt: int,
                 use_token_params: bool = True,
                 shift_size: Optional[Sequence[int]] = None,
                 attn_drop: float = 0.0,
                 proj_drop: float = 0.0,
                 use_checkpoint: bool = False,
                 ):
        super().__init__()
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.use_checkpoint = use_checkpoint
        # Bias.
        self.pe = RelativePE(
            embed_dim=pos_bias_embed_dim,
            num_heads=num_heads,
            max_abs_pos=window_size,
            max_cap_dist=window_size,
            max_prompts=max_prompts,
            tokens_per_prompt=tokens_per_prompt,
            use_token_params=use_token_params,
        )

        self.attn_norm = nn.LayerNorm(
            hidden_channels, eps=1e-6,
        )
        self.attn = WindowAttention(
            dim=hidden_channels,
            num_heads=num_heads,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
        )

        self.mlp_norm = nn.LayerNorm(
            hidden_channels, eps=1e-6,
        )
        self.mlp = nn.Linear(
            hidden_channels, hidden_channels
        )

    def forward_attn_mlp(self, x, p=None):
        b, c, h, w, d = x.shape
        window_size, shift_size = \
            tuple(self.window_size), self.get_shift_size((h, w, d))
        # Padding.
        paddings = (0, 0, 0, 0, 0, 0)
        if any([h % window_size[0] != 0,
                w % window_size[1] != 0,
                d % window_size[2] != 0]):
            paddings = [
                math.floor((window_size[0] - h % window_size[0]) / 2),
                math.ceil((window_size[0] - h % window_size[0]) / 2),
                math.floor((window_size[1] - w % window_size[1]) / 2),
                math.ceil((window_size[1] - w % window_size[1]) / 2),
                math.floor((window_size[2] - d % window_size[2]) / 2),
                math.ceil((window_size[2] - d % window_size[2]) / 2),
            ]
            # F.pad needs reversed order.
            x = F.pad(x, tuple(reversed(paddings)))
        hp, wp, dp = x.shape[2:]
        # Positional bias.
        pos_bias = self.pe(
            dim_h=window_size[0],
            dim_w=window_size[1],
            dim_d=window_size[2],
            dim_i=p.size(1) if p is not None else 0,
        ).unsqueeze(1)  # [b, 1, num_heads, n, n]
        # Shift.
        if any([size > 0 for size in shift_size]):
            x_shifted = torch.roll(
                x,
                shifts=(-shift_size[0], -shift_size[1], -shift_size[2]),
                dims=(2, 3, 4),
            )
            with torch.no_grad():
                attn_mask = get_attn_mask(
                    shape_x=(hp, wp, dp),
                    window_size=window_size,
                    shift_size=shift_size,
                    paddings=paddings,
                    device=x.device,
                )
                if p is not None:
                    # Prompt tokens are never masked.
                    attn_mask_total = torch.zeros(
                        (attn_mask.shape[0], attn_mask.shape[1],
                         attn_mask.shape[2] + p.shape[1],
                         attn_mask.shape[3] + p.shape[1]),
                        dtype=torch.float, device=x.device
                    )
                    attn_mask_total[:, :, :-p.shape[1], :-p.shape[1]] = attn_mask
                    attn_mask_total[:, :, :-p.shape[1], -p.shape[1]:] = 1.0
                else:
                    attn_mask_total = attn_mask
                attn_mask_total = rearrange(
                    attn_mask_total, 'b p n m -> b p () n m')  # [1, P, 1, N, N]
        else:
            x_shifted = x
            attn_mask_total = None
        # Attention.
        x_windowed = window_partition(x_shifted, window_size)
        if p is not None:
            # Concatenate patched sequence with tokens.
            x_windowed = torch.cat([
                rearrange(x_windowed, 'b p c h w d -> b p (h w d) c'),
                rearrange(p, 'b i c -> b () i c').expand(
                    -1, x_windowed.size(1), -1, -1),
            ], dim=2)
        else:
            x_windowed = rearrange(x_windowed, 'b p c h w d -> b p (h w d) c')
        shortcut_attn = x_windowed
        x_windowed = self.attn_norm(x_windowed)
        x_windowed = self.attn(
            q=x_windowed, k=x_windowed, v=x_windowed,
            pos_bias=pos_bias,
            mask=attn_mask_total,
        )
        x_windowed = x_windowed + shortcut_attn
        if p is not None:
            # Cut out extra features.
            x_windowed = x_windowed[:, :, :-p.size(1), :].contiguous()
        # Mlp.
        x_windowed = x_windowed + self.mlp(self.mlp_norm(x_windowed))
        x_windowed = rearrange(
            x_windowed,
            'b p (h w d) c -> b p c h w d',
            h=window_size[0],
            w=window_size[1],
            d=window_size[2],
        )
        x_shifted = window_reverse(
            x_windowed, window_size, shape_x=(hp, wp, dp))
        # Shift back.
        if any([size > 0 for size in shift_size]):
            x = torch.roll(
                x_shifted,
                shifts=(shift_size[0], shift_size[1], shift_size[2]),
                dims=(2, 3, 4),
            )
        else:
            x = x_shifted
        # Padding.
        if any([p > 0 for p in paddings]):
            x = x[
                ...,
                paddings[0]: x.shape[2] - paddings[1],
                paddings[2]: x.shape[3] - paddings[3],
                paddings[4]: x.shape[4] - paddings[5],
                ].contiguous()

        return x

    def forward(self, x, p=None):
        if self.use_checkpoint:
            x = checkpoint.checkpoint(
                self.forward_attn_mlp, x, p, use_reentrant=False)
        else:
            x = self.forward_attn_mlp(x, p)
        return x

    def get_shift_size(self, shape_x):
        shift_size = list(self.shift_size)
        for i, d in enumerate(shape_x):
            if d <= self.window_size[i]:
                shift_size[i] = 0
        return tuple(shift_size)

    # Backbone parameters.
    def named_parameters_body(self):
        return [
            *list(self.attn_norm.named_parameters()),
            *list(self.attn.named_parameters()),
            *list(self.mlp_norm.named_parameters()),
            *list(self.mlp.named_parameters()),
        ]

    def named_parameters_bias_content(self):
        return [
            *self.pe.named_parameters_bias_content(),
        ]

    def named_parameters_bias_prompt_tokens(self):
        return [
            *self.pe.named_parameters_bias_prompt_tokens(),
        ]


def window_partition(x, window_size):
    return rearrange(
        x,
        'b c (h p1) (w p2) (d p3) -> b (p1 p2 p3) c h w d',
        h=window_size[0],
        w=window_size[1],
        d=window_size[2]
    )


def window_reverse(x, window_size, shape_x):
    return rearrange(
        x,
        'b (p1 p2 p3) c h w d -> b c (h p1) (w p2) (d p3)',
        p1=shape_x[0] // window_size[0],
        p2=shape_x[1] // window_size[1],
        p3=shape_x[2] // window_size[2]
    )


def get_attn_mask(shape_x: Sequence[int],
                  window_size: Sequence[int],
                  shift_size: Sequence[int],
                  paddings: Sequence[int],
                  device: Optional[torch.device] = None):

    with torch.no_grad():
        image_mask = torch.zeros(shape_x, dtype=torch.float, device=device)
        h_slices = (
            slice(0, -window_size[0]),
            slice(-window_size[0], -shift_size[0]),
            slice(-shift_size[0], None)
        )
        w_slices = (
            slice(0, -window_size[1]),
            slice(-window_size[1], -shift_size[1]),
            slice(-shift_size[1], None)
        )
        d_slices = (
            slice(0, -window_size[2]),
            slice(-window_size[2], -shift_size[2]),
            slice(-shift_size[2], None)
        )
        # Encode each region by an int.
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                for d in d_slices:
                    image_mask[h, w, d] = cnt
                    cnt += 1

        # Encode non-padded regions differently, so paddings can't interact
        # with true content.
        if any([p > 0 for p in paddings]):
            image_mask[
                paddings[0]: shape_x[0] - paddings[1],
                paddings[2]: shape_x[1] - paddings[3],
                paddings[4]: shape_x[2] - paddings[5]
            ] = 100

        mask_windows = rearrange(
            window_partition(image_mask.unsqueeze(0).unsqueeze(0),
                             window_size).squeeze(2),
            'b p h w d -> b p (h w d)',
        )
        # [1 (B), P, (H' W' D')]
        attn_mask = mask_windows.unsqueeze(2) - mask_windows.unsqueeze(3)
        # [1 (B), P, (H' W' D'), (H' W' D')]
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(0.0)).masked_fill(attn_mask == 0, float(1.0))
        # attn_mask = (~(attn_mask == 0)).float()
        # Multiplicative mask with zeros for regions with different int
        # encoding.
    return attn_mask