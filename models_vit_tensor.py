from functools import partial
import torch
import torch.nn as nn
from util.video_vit import Attention, Block, PatchEmbed


class VisionTransformer(nn.Module):
    """Визуальный трансформер с поддержкой глобального усреднения по всему изображению"""

    def __init__(
            self,
            num_frames,
            t_patch_size,
            img_size=224,
            patch_size=16,
            in_chans=3,
            num_classes=10,
            embed_dim=768,
            depth=12,
            num_heads=12,
            mlp_ratio=4.0,
            no_qkv_bias=False,
            drop_path_rate=0.2,
            norm_layer=nn.LayerNorm,
            dropout=0,
            sep_pos_embed=False,
            cls_embed=False,
    ):
        super().__init__()
        self.sep_pos_embed = sep_pos_embed
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim, num_frames, t_patch_size)
        num_patches = self.patch_embed.num_patches
        input_size = self.patch_embed.input_size
        self.input_size = input_size
        self.cls_embed = cls_embed

        if self.cls_embed:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        if sep_pos_embed:
            self.pos_embed_spatial = nn.Parameter(torch.zeros(1, input_size[1] * input_size[2], embed_dim))
            self.pos_embed_temporal = nn.Parameter(torch.zeros(1, input_size[0], embed_dim))
            if self.cls_embed:
                self.pos_embed_class = nn.Parameter(torch.zeros(1, 1, embed_dim))
        else:
            if self.cls_embed:
                _num_patches = num_patches + 1
            else:
                _num_patches = num_patches
            self.pos_embed = nn.Parameter(torch.zeros(1, _num_patches, embed_dim), requires_grad=True)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]

        self.blocks = nn.ModuleList(
            [
                Block(
                    embed_dim,
                    num_heads,
                    mlp_ratio,
                    qkv_bias=not no_qkv_bias,
                    qk_scale=None,
                    norm_layer=norm_layer,
                    drop_path=dpr[i],
                    attn_func=partial(Attention, input_size=self.patch_embed.input_size, ),
                )
                for i in range(depth)
            ]
        )

        self.norm = norm_layer(embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(embed_dim, num_classes)

        torch.nn.init.normal_(self.head.weight, std=0.02)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {
            "cls_token",
            "pos_embed",
            "pos_embed_spatial",
            "pos_embed_temporal",
            "pos_embed_class",
        }

    def forward(self, x):
        x = torch.unsqueeze(x, dim=1)
        x = self.patch_embed(x)
        N, T, L, C = x.shape  # T: временная ось; L: пространственная ось

        x = x.view([N, T * L, C])

        if self.cls_embed:
            cls_token = self.cls_token
            cls_tokens = cls_token.expand(x.shape[0], -1, -1)
            x = torch.cat((cls_tokens, x), dim=1)

        if self.sep_pos_embed:
            pos_embed = (self.pos_embed_spatial.repeat(1, self.input_size[0], 1) + torch.repeat_interleave(self.pos_embed_temporal, self.input_size[1] * self.input_size[2], dim=1, ))
            if self.cls_embed:
                pos_embed = torch.cat([self.pos_embed_class.expand(pos_embed.shape[0], -1, -1), pos_embed, ], 1, )
        else:
            pos_embed = self.pos_embed[:, :, :]
        x = x + pos_embed
        requires_t_shape = (
                len(self.blocks) > 0  # поддержка пустого декодера
                and hasattr(self.blocks[0].attn, "requires_t_shape")
                and self.blocks[0].attn.requires_t_shape
        )
        if requires_t_shape:
            x = x.view([N, T, L, C])

        # применяем блоки трансформера
        for blk in self.blocks:
            x = blk(x)
        if requires_t_shape:
            x = x.view([N, T * L, C])

        # классификатор
        x = x[:, 1:, :].mean(dim=1)  # глобальный пуллинг
        x = self.norm(x)
        x = self.dropout(x)
        x = self.head(x)

        return x


def vit_base_patch8_128(**kwargs):
    model = VisionTransformer(
        img_size=128,
        in_chans=1,
        patch_size=8,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        num_frames=12,
        t_patch_size=3,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )
    return model
