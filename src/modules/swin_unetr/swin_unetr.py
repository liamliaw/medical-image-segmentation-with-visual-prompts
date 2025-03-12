from monai.networks.blocks import UnetrBasicBlock, UnetrUpBlock
import torch
import torch.nn as nn
from .unet_blocks import SwinUpBlock
from ..swin_transformer import ConsecutiveSwinBlocks


class SwinUnetR(nn.Module):
    def __init__(self, conf):
        super().__init__()
        self.input_layer = None
        self.encoder_blocks = None
        self.bottleneck = None
        self.residual_blocks = None
        self.decoder_blocks = None
        self.output_layer = None
        self.prompt_tokens = nn.ModuleDict()
        self.extra_heads = nn.ModuleDict()
        self.conf = conf

        if conf.training_mode == 'self_supervised_learning_encoder':
            self.setup_ssl_encoder()
        elif (conf.training_mode == 'self_supervised_learning_decoder'
              or conf.training_mode == 'supervised_learning_decoder'):
            self.setup_ssl_decoder()
            params = self.named_parameters_encoder(
                include_prompt_tokens=conf.use_encoder_prompting)
            for (_, p) in params:
                p.requires_grad = False
        elif (conf.training_mode == 'self_supervised_learning_all'
              or conf.training_mode == 'supervised_learning_all'):
            self.setup_ssl_decoder()
        elif conf.training_mode == 'downstream':
            self.setup_downstream()
            for (n, p) in self.named_parameters_encoder(
                    include_prompt_tokens=False):
                p.requires_grad = False
            for (n, p) in self.named_parameters_decoder(
                    include_prompt_tokens=False):
                p.requires_grad = False

        else:
            raise ValueError(
                f'Training mode {conf.training_mode} not available!')

    def forward_swin_transformer(self, x):
        out_list = []
        out_list.insert(0, x)
        enc = self.input_layer(x)
        out_list.insert(0, enc)
        for j in range(self.conf.depth_unet):
            if not self.conf.use_encoder_prompting:
                p_w, p_sw = None, None
            else:
                # Get prompt tokens.
                p_w, p_sw = self.prompt_tokens['enc'][2 * j], \
                    self.prompt_tokens['enc'][2 * j + 1]
                # Broadcast tokens.
                p_w, p_sw = p_w.unsqueeze(0).repeat(enc.size(0), 1, 1), \
                    p_sw.unsqueeze(0).repeat(enc.size(0), 1, 1)
            enc = self.encoder_blocks[j](enc, [p_w, p_sw])
            out_list.insert(0, enc)
        return {'out_vit': out_list}

    def forward_ssl_encoder(self, x):
        output_dict = {}
        out_vit = self.forward_swin_transformer(x)['out_vit']
        # Proxy task heads.
        if self.conf.training_mode == 'self_supervised_learning_encoder':
            if self.conf.use_reconstruction or self.conf.use_mutual_learning:
                output_dict['reconstruction'] = \
                    self.extra_heads['reconstruction'](out_vit[0])
            if self.conf.use_rotation_prediction:
                x_pool = nn.AdaptiveAvgPool3d((1, 1, 1))(
                    out_vit[0]).squeeze(-1).squeeze(-1).squeeze(-1)
                output_dict['rotation_prediction'] = \
                    self.extra_heads['rotation_prediction'](x_pool)
            if self.conf.use_contrastive_learning:
                x_pool = nn.AdaptiveAvgPool3d((1, 1, 1))(
                        out_vit[0]).squeeze(-1).squeeze(-1).squeeze(-1)
                output_dict['contrastive_coding'] = \
                    self.extra_heads['contrastive_coding'](x_pool)
        output_dict['out_vit'] = out_vit
        return output_dict

    def forward_decoder(self, c):
        b = self.bottleneck(c[0]) + c[0]
        dec = b
        for j in range(self.conf.depth_unet):
            if not self.conf.use_decoder_prompting:
                p_w, p_sw = None, None
            else:
                p_w, p_sw = self.prompt_tokens['dec'][2 * j], \
                    self.prompt_tokens['dec'][2 * j + 1]
                p_w, p_sw = p_w.unsqueeze(0).repeat(dec.size(0), 1, 1), \
                    p_sw.unsqueeze(0).repeat(dec.size(0), 1, 1)
            res = self.residual_blocks[j](c[j + 1])
            dec = self.decoder_blocks[j](dec, res, [p_w, p_sw])
        if self.conf.unetr_res_block == 'none':
            out = self.output_layer(dec)
        else:
            # With residual blocks.
            if not self.conf.use_decoder_prompting:
                p_w, p_sw = None, None
            else:
                p_w, p_sw = self.prompt_tokens['out'][0], \
                    self.prompt_tokens['out'][1]
                p_w, p_sw = p_w.unsqueeze(0).repeat(dec.size(0), 1, 1), \
                    p_sw.unsqueeze(0).repeat(dec.size(0), 1, 1)
            out = self.output_layer(
                dec, self.residual_blocks[-1](c[-1]), [p_w, p_sw])
        return {'latent_outputs': out}

    def forward_ssl_decoder(self, x):
        out_enc = self.forward_ssl_encoder(x)
        out_dec = self.forward_decoder(out_enc['out_vit'])
        if (self.conf.training_mode == 'supervised_learning_decoder'
            or self.conf.training_mode == 'supervised_learning_all'):
            out_dec['seg_pred'] = self.extra_heads['segmentation'](
                out_dec['latent_outputs'])
        return out_dec

    def forward_downstream(self, x):
        out_dec = self.forward_ssl_decoder(x)
        # Segmentation head.
        seg = self.extra_heads['downstream'](out_dec['latent_outputs'])
        return {'downstream': seg}

    def forward(self, x):
        if self.conf.training_mode == 'self_supervised_learning_encoder':
            output_dict = self.forward_ssl_encoder(x)
            return output_dict
        elif (self.conf.training_mode == 'self_supervised_learning_decoder'
              or self.conf.training_mode == 'self_supervised_learning_all'
              or self.conf.training_mode == 'supervised_learning_decoder'
              or self.conf.training_mode == 'supervised_learning_all'):
            output_dict = self.forward_ssl_decoder(x)
            return output_dict
        elif self.conf.training_mode == 'downstream':
            output_dict = self.forward_downstream(x)
            return output_dict
        else:
            raise ValueError(
                f'Training mode {self.conf.training_mode} not available!')

    def setup_swin_transformer(self, in_chs):
        # Patch embedding.
        self.input_layer = nn.Sequential(
            nn.Conv3d(
                in_channels=self.conf.input_channels,
                out_channels=self.conf.hidden_channels[0],
                kernel_size=self.conf.input_patch_size,
                stride=self.conf.input_patch_size,
            ),
            nn.BatchNorm3d(
                num_features=self.conf.hidden_channels[0], eps=1e-6,
            ),
        )
        # Hardcoded.
        merge_last_dim = [True if i < 1 else False
                          for i in range(self.conf.depth_unet)]
        # Swin encoder blocks.
        self.encoder_blocks = nn.ModuleList([
            ConsecutiveSwinBlocks(
                hidden_channels=in_chs[i],
                pos_bias_embed_dim=self.conf.pos_bias_embed_dim,
                num_heads=self.conf.num_heads_encoder * (2 ** i),
                window_size=self.conf.attn_window_size,
                max_prompts=self.conf.max_prompts,
                tokens_per_prompt=self.conf.tokens_per_prompt_encoder,
                use_token_params=self.conf.use_encoder_prompting,
                down=True,
                merge_last_dim=merge_last_dim[i],
                attn_drop=self.conf.attn_drop,
                proj_drop=self.conf.proj_drop,
                use_checkpoint=self.conf.use_checkpoint,
            ) for i in range(self.conf.depth_unet)
        ])

    def setup_ssl_encoder(self):
        in_chs = [self.conf.hidden_channels[i]
                  for i in range(self.conf.depth_unet)]
        self.setup_swin_transformer(in_chs=in_chs)
        # Setup proxy task heads.
        if self.conf.use_reconstruction or self.conf.use_mutual_learning:
            reconstruction_head = nn.ModuleList()
            rec_chs = [self.conf.hidden_channels[-1] // (2 ** i)
                       for i in range(len(in_chs) + 1)] \
                + [self.conf.hidden_channels[-1] // (2 ** len(in_chs))]
            scale_depth = [1 if i < len(in_chs) - 1 else 2
                            for i in range(len(in_chs) + 1)]
            for i in range(len(in_chs) + 1):
                reconstruction_head.extend([
                    nn.Conv3d(
                        in_channels=rec_chs[i], out_channels=rec_chs[i + 1],
                        kernel_size=3, stride=1, padding=1,
                    ),
                    nn.InstanceNorm3d(rec_chs[i + 1]),
                    nn.LeakyReLU(),
                    nn.Upsample(
                        scale_factor=(2, 2, scale_depth[i]), mode='trilinear', align_corners=True,
                    ),
                ])
            reconstruction_head.extend([
                nn.Conv3d(
                    in_channels=rec_chs[-1],
                    out_channels=self.conf.input_channels,
                    kernel_size=1, stride=1,
                ),
            ])
            self.extra_heads['reconstruction'] = nn.Sequential(
                *reconstruction_head)
        if self.conf.use_rotation_prediction:
            self.extra_heads['rotation_prediction'] = nn.Linear(
                in_features=self.conf.hidden_channels[-1],
                out_features=4,
            )
        if self.conf.use_contrastive_learning:
            self.extra_heads['contrastive_coding'] = nn.Linear(
                in_features=self.conf.hidden_channels[-1],
                out_features=self.conf.contrastive_coding_dim,
            )
        if self.conf.use_encoder_prompting:
            self.setup_prompt_tokens_encoder()

    def setup_downstream(self):
        self.setup_ssl_decoder()
        # Segmentation head.
        self.extra_heads['downstream'] = nn.Sequential(
            nn.BatchNorm3d(
                num_features=self.conf.hidden_channels[0], ),
            nn.Conv3d(
                in_channels=self.conf.hidden_channels[0],
                out_channels=self.conf.output_channels_downstream,
                kernel_size=3, stride=1, padding=1,
            )
        )

    def setup_ssl_decoder(self):
        in_chs = [self.conf.hidden_channels[i]
                  for i in range(self.conf.depth_unet)]
        out_chs = [self.conf.hidden_channels[i + 1]
                   for i in range(self.conf.depth_unet)]
        self.setup_swin_transformer(in_chs=in_chs)
        in_chs.reverse(), out_chs.reverse()
        # Decoders.
        # Bottleneck.
        if self.conf.unetr_res_block == 'full':
            self.bottleneck = UnetrBasicBlock(
                spatial_dims=3,
                in_channels=out_chs[0],
                out_channels=out_chs[0],
                kernel_size=3,
                stride=1,
                norm_name='instance',
                res_block=self.conf.basic_block_res,
            )
        else:
            # Simplified bottleneck.
            self.bottleneck = nn.Conv3d(
                in_channels=out_chs[0],
                out_channels=out_chs[0],
                kernel_size=3,
                stride=1,
                padding=1,
            )
        # Residual blocks.
        if self.conf.unetr_res_block == 'full':
            self.residual_blocks = nn.ModuleList([
                UnetrBasicBlock(
                    spatial_dims=3,
                    in_channels=in_chs[i],
                    out_channels=in_chs[i],
                    kernel_size=3,
                    stride=1,
                    norm_name='instance',
                    res_block=self.conf.basic_block_res,
                ) for i in range(self.conf.depth_unet)
            ])
            self.residual_blocks.append(
                UnetrBasicBlock(
                    spatial_dims=3,
                    in_channels=self.conf.input_channels,
                    out_channels=in_chs[-1],
                    kernel_size=3,
                    stride=1,
                    norm_name='instance',
                    res_block=self.conf.basic_block_res,
            ))
        elif self.conf.unetr_res_block == 'simple':
            # Simplified residual blocks.
            self.residual_blocks = nn.ModuleList([
                nn.Conv3d(
                    in_channels=in_chs[i],
                    out_channels=in_chs[i],
                    kernel_size=3,
                    stride=1,
                    padding=1,
                ) for i in range(self.conf.depth_unet)
            ])
            self.residual_blocks.append(
                nn.Conv3d(
                    in_channels=self.conf.input_channels,
                    out_channels=in_chs[-1],
                    kernel_size=3,
                    stride=1,
                    padding=1,
            ))
        else:
            # No residual block.
            self.residual_blocks = nn.ModuleList([
                nn.Identity() for i in range(self.conf.depth_unet + 1)
            ])
        # Hardcoded.
        scale_depth = [1 if i < len(in_chs) - 1 else 2
                       for i in range(self.conf.depth_unet)]
        # Up blocks.
        if self.conf.unetr_up_block == 'swin':
            self.decoder_blocks = nn.ModuleList([
                SwinUpBlock(
                    in_channels=out_chs[i],
                    out_channels=in_chs[i],
                    strides=(2, 2, scale_depth[i]),
                    kernel_size=(3, 3, 3),
                    pos_bias_embed_dim=self.conf.pos_bias_embed_dim,
                    num_heads=self.conf.num_heads_decoder,
                    window_size=self.conf.attn_window_size,
                    max_prompts=self.conf.max_prompts,
                    tokens_per_prompt=self.conf.tokens_per_prompt_decoder,
                    use_token_params=self.conf.use_decoder_prompting,
                    attn_drop=self.conf.attn_drop,
                    proj_drop=self.conf.proj_drop,
                    use_checkpoint=self.conf.use_checkpoint,
                ) for i in range(self.conf.depth_unet)
            ])
        else:
            # CNN based decoder blocks.
            self.decoder_blocks = nn.ModuleList([
                UnetrUpBlock(
                    spatial_dims=3,
                    in_channels=out_chs[i],
                    out_channels=in_chs[i],
                    kernel_size=3,
                    upsample_kernel_size=2,
                    norm_name='instance',
                    res_block=self.conf.res_block,
                ) for i in range(self.conf.depth_unet)
            ])
        # Output layer.
        if self.conf.unetr_res_block == 'none':
            self.output_layer = nn.Upsample(
                scale_factor=(2, 2, 2),
                mode='trilinear',
                align_corners=False,
            )
        else:
            self.output_layer = SwinUpBlock(
                in_channels=in_chs[-1],
                out_channels=in_chs[-1],
                hidden_channels=2 * in_chs[-1],
                strides=(2, 2, 2),
                kernel_size=(3, 3, 3),
                pos_bias_embed_dim=self.conf.pos_bias_embed_dim,
                num_heads=self.conf.num_heads_decoder,
                window_size=self.conf.attn_window_size,
                max_prompts=self.conf.max_prompts,
                tokens_per_prompt=self.conf.tokens_per_prompt_decoder,
                attn_drop=self.conf.attn_drop,
                proj_drop=self.conf.proj_drop,
                use_checkpoint=self.conf.use_checkpoint,
            ) if self.conf.unetr_up_block == 'swin' else \
                UnetrUpBlock(
                    spatial_dims=3,
                    in_channels=in_chs[-1],
                    out_channels=in_chs[-1],
                    kernel_size=3,
                    upsample_kernel_size=2,
                    norm_name='instance',
                    res_block=self.conf.res_block,
                )

        if (self.conf.training_mode == 'supervised_learning_decoder'
            or self.conf.training_mode == 'supervised_learning_all'):
            # Segmentation head for supervised pretraining.
            self.extra_heads['segmentation'] = nn.Sequential(
                nn.BatchNorm3d(
                    num_features=self.conf.hidden_channels[0],),

                nn.Conv3d(
                    in_channels=self.conf.hidden_channels[0],
                    out_channels=self.conf.output_channels_pretrain,
                    kernel_size=3, stride=1, padding=1,
                )
            )
        if self.conf.use_encoder_prompting:
            self.setup_prompt_tokens_encoder()
        if self.conf.use_decoder_prompting:
            self.setup_prompt_tokens_decoder()

    def setup_prompt_tokens_encoder(self):
        hidden_chs = self.conf.hidden_channels
        if self.conf.use_encoder_prompting:
            self.prompt_tokens['enc'] = nn.ParameterList([
                nn.Parameter(nn.init.xavier_uniform_(
                    torch.empty((self.conf.tokens_per_prompt_encoder,
                                 hidden_chs[i // 2])),
                    gain=nn.init.calculate_gain('linear')),
                    requires_grad=True,
                ) for i in range(2 * self.conf.depth_unet)
            ])

    def setup_prompt_tokens_decoder(self):
        hidden_chs = self.conf.hidden_channels
        if self.conf.use_decoder_prompting:
            self.prompt_tokens['dec'] = nn.ParameterList([
                nn.Parameter(nn.init.xavier_uniform_(
                    torch.empty((self.conf.tokens_per_prompt_decoder,
                                 hidden_chs[-(i + 1) // 2 - 1])),
                    gain=nn.init.calculate_gain('linear')),
                    requires_grad=True,
                ) for i in range(2 * self.conf.depth_unet)
            ])
            if self.conf.unetr_res_block != 'none' \
                and self.conf.unetr_up_block == 'swin':
                self.prompt_tokens['out'] = nn.ParameterList([
                    nn.Parameter(nn.init.xavier_uniform_(
                        torch.empty((self.conf.tokens_per_prompt_decoder, hidden_chs[0])),
                        gain=nn.init.calculate_gain('linear')),
                        requires_grad=True,
                    ) for i in range(2)
                ])

    # Downstream trainable parameters.
    def named_parameters_downstream(self):
        params = []
        if self.conf.use_encoder_prompting:
            params.extend(self.named_parameters_prompt_tokens_encoder())
        if self.conf.use_decoder_prompting:
            params.extend(self.named_parameters_prompt_tokens_decoder())
        params.extend(self.extra_heads['downstream'].named_parameters())
        return params

    # Encoder prompt tokens parameters.
    def named_parameters_prompt_tokens_encoder(self):
        params_tokens = [(n, p) for (n, p) in self.prompt_tokens['enc'].named_parameters()]
        params_bias = []
        for i in range(self.conf.depth_unet):
            params_bias.extend(
                self.encoder_blocks[i].named_parameters_bias_prompt_tokens(),
            )
        return [
            *params_tokens,
            *params_bias,
        ]

    # Decoder prompt tokens parameters.
    def named_parameters_prompt_tokens_decoder(self):
        params_tokens = [(n, p) for (n, p) in self.prompt_tokens['dec'].named_parameters()]
        params_bias = []
        for i in range(self.conf.depth_unet):
            params_bias.extend(
                self.decoder_blocks[i].named_parameters_bias_prompt_tokens(),
            )
        if self.conf.unetr_res_block != 'none' \
                and self.conf.unetr_up_block == 'swin':
            params_tokens.extend(
                [(n, p) for (n, p) in self.prompt_tokens['out'].named_parameters()])
        if self.conf.unetr_res_block != 'none':
            params_bias.extend(self.output_layer.named_parameters_bias_prompt_tokens())

        return [
            *params_tokens,
            *params_bias,
        ]

    # Backbone encoder parameters.
    def named_parameters_encoder(self, include_prompt_tokens=False):
        params_swin = []
        for i in range(self.conf.depth_unet):
            params_swin.extend(
                self.encoder_blocks[i].named_parameters_body())
            params_swin.extend(
                self.encoder_blocks[i].named_parameters_bias_content())
        params = [
            *list(self.input_layer.named_parameters()),
            *params_swin,
        ]
        if include_prompt_tokens and self.conf.use_encoder_prompting:
            params_tokens = self.named_parameters_prompt_tokens_encoder()
            params.extend(params_tokens)
        if self.conf.training_mode == 'self_supervised_learning_encoder':
            for _, head in self.extra_heads.items():
                params.extend(head.named_parameters())
        return params

    # Backbone decoder parameters.
    def named_parameters_decoder(self, include_prompt_tokens=False):
        params_res = []
        for i in range(self.conf.depth_unet + 1):
            params_res.extend(
                self.residual_blocks[i].named_parameters()
            )
        params_decoder = []
        for i in range(self.conf.depth_unet):
            params_decoder.extend(
                self.decoder_blocks[i].named_parameters_body())
            params_decoder.extend(
                self.decoder_blocks[i].named_parameters_bias_content())
        params = [
            *list(self.bottleneck.named_parameters()),
            *params_res,
            *params_decoder,
            # *list(self.output_layer.named_parameters_body()),
            # *list(self.output_layer.named_parameters_bias_content())
        ]
        if self.conf.unetr_res_block != 'none':
            params.extend(self.output_layer.named_parameters_body())
            params.extend(self.output_layer.named_parameters_bias_content())
        if include_prompt_tokens and self.conf.use_decoder_prompting:
            params_tokens = self.named_parameters_prompt_tokens_decoder()
            params.extend(params_tokens)
        if (self.conf.training_mode == 'supervised_learning_decoder'
            or self.conf.training_mode == 'supervised_learning_all'):
            params.extend(self.extra_heads['segmentation'].named_parameters())
        # if self.conf.training_mode == 'downstream':
        #     params.extend(self.extra_heads['downstream'].named_parameters())
        return params