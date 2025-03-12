from einops import rearrange
from monai.networks.layers.utils import get_act_layer, get_norm_layer
from monai.networks.blocks import Convolution
import torch
import torch.nn as nn
from typing import Sequence, Iterator, Tuple
from ..swin_transformer import ConsecutiveSwinBlocks


# Decoder block.
class SwinUpBlock(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 strides: Sequence[int],
                 kernel_size: Sequence[int],
                 pos_bias_embed_dim: int,
                 num_heads: int,
                 window_size: Sequence[int],
                 max_prompts: int,
                 tokens_per_prompt: int,
                 use_token_params: bool = True,
                 act: str = "leakyrelu",
                 norm: str = "batch",
                 attn_drop: float = 0.0,
                 proj_drop: float = 0.0,
                 use_checkpoint: bool = False,
                 hidden_channels=None,
                 ):
        super().__init__()
        self.up = nn.Upsample(
            scale_factor=tuple(strides),
            mode='trilinear',
            align_corners=False,
        )
        self.act = get_act_layer(
            name=act
        )
        if hidden_channels is None:
            hidden_channels = in_channels + in_channels // 2
        self.norm_concat = get_norm_layer(
            name=norm,
            spatial_dims=3,
            channels=hidden_channels,
        )
        self.conv_concat = Convolution(
            spatial_dims=3,
            in_channels=hidden_channels,
            out_channels=out_channels,
            strides=(1, 1, 1),
            kernel_size=kernel_size,
            act="leakyrelu",
            norm="batch",
            conv_only=True,
            is_transposed=False,
        )
        self.swin_layer = ConsecutiveSwinBlocks(
            hidden_channels=out_channels,
            pos_bias_embed_dim=pos_bias_embed_dim,
            num_heads=num_heads,
            window_size=window_size,
            max_prompts=max_prompts,
            tokens_per_prompt=tokens_per_prompt,
            use_token_params=use_token_params,
            down=False,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            use_checkpoint=use_checkpoint,
        )

    def forward(self, x, c, p=(None, None)):
        x = self.up(x)
        x = torch.cat([x[..., :c.size(2), :c.size(3), :c.size(4)], c], dim=1)
        x = self.conv_concat(self.act(self.norm_concat(x)))
        x = self.swin_layer(x, p)
        return x

    # Get backbone parameters.
    def named_parameters_body(self):
        return [
            *self.norm_concat.named_parameters(),
            *self.conv_concat.named_parameters(),
            *self.swin_layer.named_parameters_body(),
        ]

    # Get image content bias.
    def named_parameters_bias_content(self):
        return self.swin_layer.named_parameters_bias_content()

    # Get prompt tokens bias.
    def named_parameters_bias_prompt_tokens(self):
        return self.swin_layer.named_parameters_bias_prompt_tokens()