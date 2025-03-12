import torch
from einops import rearrange
import torch.nn as nn
from typing import Sequence


class RelativePE(nn.Module):
    def __init__(self,
                 embed_dim: int,
                 num_heads: int,
                 max_abs_pos: Sequence[int],
                 max_cap_dist: Sequence[int],
                 max_prompts: int,
                 tokens_per_prompt: int,
                 use_token_params: bool = True,
                 ):
        super().__init__()
        self.scale = embed_dim ** -0.5
        self.num_heads = num_heads
        # Content parameters.
        self.enc_content_h = nn.Parameter(
            nn.init.xavier_uniform_(
                torch.empty((2 * max_cap_dist[0] - 1, embed_dim)),
                gain=nn.init.calculate_gain('linear')),
            requires_grad=True,
        )
        self.enc_content_w = nn.Parameter(
            nn.init.xavier_uniform_(
                torch.empty((2 * max_cap_dist[1] - 1, embed_dim)),
                gain=nn.init.calculate_gain('linear')),
            requires_grad=True,
        )
        self.enc_content_d = nn.Parameter(
            nn.init.xavier_uniform_(
                torch.empty((2 * max_cap_dist[2] - 1, embed_dim)),
                gain=nn.init.calculate_gain('linear')),
            requires_grad=True,
        )

        relative_dist_h = (
            torch.arange(max_abs_pos[0], dtype=torch.long).reshape(1, -1)
            - torch.arange(max_abs_pos[0], dtype=torch.long).reshape(-1, 1)
        )
        relative_dist_w = (
            torch.arange(max_abs_pos[1], dtype=torch.long).reshape(1, -1)
            - torch.arange(max_abs_pos[1], dtype=torch.long).reshape(-1, 1)
        )
        relative_dist_d = (
            torch.arange(max_abs_pos[2], dtype=torch.long).reshape(1, -1)
            - torch.arange(max_abs_pos[2], dtype=torch.long).reshape(-1, 1)
        )

        relative_dist_h = torch.clamp(relative_dist_h + max_cap_dist[0] - 1,
                                      min=0, max=(max_cap_dist[0] - 1) * 2)
        relative_dist_w = torch.clamp(relative_dist_w + max_cap_dist[1] - 1,
                                      min=0, max=(max_cap_dist[1] - 1) * 2)
        relative_dist_d = torch.clamp(relative_dist_d + max_cap_dist[2] - 1,
                                      min=0, max=(max_cap_dist[2] - 1) * 2)

        self.register_buffer('relative_dist_h', relative_dist_h)
        self.register_buffer('relative_dist_w', relative_dist_w)
        self.register_buffer('relative_dist_d', relative_dist_d)
        # Weights for bias.
        self.weights_content_h = nn.Parameter(
            nn.init.xavier_uniform_(
                torch.empty((num_heads, embed_dim)),
                gain=nn.init.calculate_gain('linear')),
            requires_grad=True,
        )
        self.weights_content_w = nn.Parameter(
            nn.init.xavier_uniform_(
                torch.empty((num_heads, embed_dim)),
                gain=nn.init.calculate_gain('linear')),
            requires_grad=True,
        )
        self.weights_content_d = nn.Parameter(
            nn.init.xavier_uniform_(
                torch.empty((num_heads, embed_dim)),
                gain=nn.init.calculate_gain('linear')),
            requires_grad=True,
        )
        # Token parameters.
        if use_token_params:
            self.enc_token = nn.ParameterList([
                nn.Parameter(
                    nn.init.xavier_uniform_(
                        torch.empty((tokens_per_prompt, embed_dim)),
                        gain=nn.init.calculate_gain('linear')),
                    requires_grad=True,
                ) for _ in range(max_prompts)
            ])
            self.weights_token = nn.Parameter(
                nn.init.xavier_uniform_(
                    torch.empty((num_heads, embed_dim)),
                    gain=nn.init.calculate_gain('linear')),
                requires_grad=True,
            )

    def forward(self, dim_h, dim_w, dim_d, dim_i=0):
        # Content scores.
        h_emb = self.enc_content_h[self.relative_dist_h[:dim_h, :dim_h], :]
        w_emb = self.enc_content_w[self.relative_dist_w[:dim_w, :dim_w], :]
        d_emb = self.enc_content_d[self.relative_dist_d[:dim_d, :dim_d], :]
        row_scores = torch.einsum(
            'h c, n m c -> h n m',
            self.weights_content_h, h_emb,
        ).unsqueeze(0)
        col_scores = torch.einsum(
            'h c, n m c -> h n m',
            self.weights_content_w, w_emb,
        ).unsqueeze(0)
        dep_scores = torch.einsum(
            'h c, n m c -> h n m',
            self.weights_content_d, d_emb,
        ).unsqueeze(0)  # [1, num_heads, d, d]
        content_scores = (
            rearrange(row_scores, 'b h n m -> b h n () () m () ()')
            + rearrange(col_scores, 'b h n m -> b h () n () () m ()')
            + rearrange(dep_scores, 'b h n m -> b h () () n () () m')
        ) / 3  # [1, num_heads, h, w, d, h, w, d]
        content_scores = rearrange(
            content_scores, 'b h i j k l m n -> b h (i j k) (l m n)'
        ) * self.scale
        if dim_i == 0:
            return content_scores
        else:
            # Token scores.
            total_scores = torch.zeros(
                (content_scores.size(0), self.num_heads,
                 dim_h * dim_w * dim_d + dim_i, dim_h * dim_w * dim_d + dim_i),
                dtype=torch.float,
                device=self.weights_token.device,
            )
            token_emb = torch.cat(list(self.enc_token)).unsqueeze(0)
            token_scores = torch.einsum(
                'h c, b n c -> b h n',
                self.weights_token, token_emb,
            ) * self.scale
            total_scores[:, :, :-dim_i, :-dim_i] = content_scores
            total_scores[:, :, :-dim_i, -dim_i:] = \
                token_scores.unsqueeze(-2).expand(-1, -1, dim_h * dim_w * dim_d, -1)
            return total_scores

    def named_parameters_bias_content(self):
        return [
            (name, param) for name, param in self.named_parameters()
            if any([n in name for n in ['enc_content', 'weights_content']])
        ]

    def named_parameters_bias_prompt_tokens(self):
        return [
            (name, param) for name, param in self.named_parameters()
            if any([s in name for s in ['enc_token', 'weights_token']])
        ]