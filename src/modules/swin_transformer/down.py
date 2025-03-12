from einops import rearrange
import torch
import torch.nn as nn


class PatchMerging(nn.Module):
    def __init__(self, in_channels, out_channels, merge_last_dim=True):
        super().__init__()
        self.norm = nn.LayerNorm(
            (8 * in_channels) if merge_last_dim else (4 * in_channels),
            eps=1e-6,
        )
        self.reduction = nn.Linear(
            in_features=(8 * in_channels) if merge_last_dim
            else (4 * in_channels),
            out_features=out_channels,
            bias=False,
        )
        self.merge_last_dim = merge_last_dim

    def forward(self, x):
        b, c, h, w, d = x.shape
        pad_h = 0 if h % 2 == 0 else 1
        pad_w = 0 if w % 2 == 0 else 1
        pad_d = 0 if d % 2 == 0 else 1
        paddings = (0, pad_h, 0, pad_w, 0, pad_d)
        if pad_h == 1 or pad_w == 1 or pad_d == 1:
            x = nn.functional.pad(x, tuple(reversed(paddings)))
        _, _, h, w, d = x.shape
        if self.merge_last_dim:
            x0 = x[:, :, 0::2, 0::2, 0::2]
            x1 = x[:, :, 1::2, 0::2, 0::2]
            x2 = x[:, :, 0::2, 1::2, 0::2]
            x3 = x[:, :, 0::2, 0::2, 1::2]
            x4 = x[:, :, 1::2, 1::2, 0::2]
            x5 = x[:, :, 1::2, 0::2, 1::2]
            x6 = x[:, :, 0::2, 1::2, 1::2]
            x7 = x[:, :, 1::2, 1::2, 1::2]
            x = torch.cat([x0, x1, x2, x3, x4, x5, x6, x7], dim=1)
        else:
            x0 = x[:, :, 0::2, 0::2, :]
            x1 = x[:, :, 1::2, 0::2, :]
            x2 = x[:, :, 0::2, 1::2, :]
            x3 = x[:, :, 1::2, 1::2, :]
            x = torch.cat([x0, x1, x2, x3], dim=1)
        x = rearrange(x, 'b c h w d -> b (h w d) c')
        x = self.reduction(self.norm(x))
        x = rearrange(
            x,
            'b (h w d) c -> b c h w d',
            h=h // 2, w=w // 2, d=d // 2 if self.merge_last_dim else d,
        )
        return x

    # Backbone parameters.
    def named_parameters_body(self):
        return [
            *list(self.reduction.named_parameters()),
            *list(self.norm.named_parameters()),
        ]