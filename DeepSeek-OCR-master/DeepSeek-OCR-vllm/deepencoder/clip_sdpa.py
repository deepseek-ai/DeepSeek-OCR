from contextlib import nullcontext
import math
from typing import Optional, Tuple
# from megatron.model import LayerNorm
from easydict import EasyDict as adict
import torch
from torch.nn import functional as F
from torch import nn
from flash_attn import flash_attn_qkvpacked_func, flash_attn_func
# from optimus import flash_attn_func
# from megatron.core import tensor_parallel
# from megatron.core import parallel_state as mpu
# from megatron.core.utils import make_viewless_tensor, divide
# from megatron.model.fused_rms_norm import RMSNorm
# from megatron.model.transformer import (
#     FlashSelfAttention,
#     NoopTransformerLayer,
#     _cfg_to_kwargs,
# )
# from megatron.model.enums import AttnMaskType, AttnType
# from megatron.model.fused_softmax import FusedScaleMaskSoftmax
# from megatron.model.utils import attention_mask_func

# from megatron.model.module import MegatronModule

# try:
#     from einops import rearrange
# except ImportError:
#     rearrange = None

# from flash_attn import flash_attn_varlen_func as flash_attn_unpadded_func

# try:
#     # flash attention 2.x
#     from flash_attn import flash_attn_varlen_func as flash_attn_unpadded_func
# except ImportError:
#     try:
#         # flash attention 1.x
#         from flash_attn.flash_attn_interface import flash_attn_unpadded_func
#     except ImportError:
#         flash_attn_unpadded_func = None

# try:
#     from flash_attn.flash_attn_interface import flash_attn_unpadded_relative_attention_bias_func
# except ImportError:
#     flash_attn_unpadded_relative_attention_bias_func = None

# try:
#     from flash_attn.flash_attn_interface import mask_flash_attn_unpadded_func
# except ImportError:
#     mask_flash_attn_unpadded_func = None


class LayerNormfp32(torch.nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


def get_abs_pos(abs_pos, tgt_size):
        """
    Resize absolute positional embeddings to target size if necessary.

    Args:
        abs_pos (torch.Tensor): [1, H, W, C] absolute positional embeddings
        tgt_size (int): target height/width

    Returns:
        torch.Tensor: resized absolute positional embeddings [1, tgt_size, tgt_size, C]
    """
    # abs_pos: L, C
    # tgt_size: M
    # return: M, C

    # print(tgt_size)
    # print(abs_pos.shape)
    # exit()
    dim = abs_pos.size(-1)
    # print(dim)
    abs_pos_new = abs_pos.squeeze(0)
    cls_token, old_pos_embed = abs_pos_new[:1], abs_pos_new[1:]



    src_size = int(math.sqrt(abs_pos_new.shape[0] - 1))
    tgt_size = int(math.sqrt(tgt_size))
    dtype = abs_pos.dtype

    if src_size != tgt_size:
        old_pos_embed = old_pos_embed.view(1, src_size, src_size, dim).permute(0, 3, 1,
                                                                                    2).contiguous()
        old_pos_embed = old_pos_embed.to(torch.float32)
        new_pos_embed = F.interpolate(
            old_pos_embed,
            size=(tgt_size, tgt_size),
            mode='bicubic',
            antialias=True,
            align_corners=False,
        ).to(dtype)
        new_pos_embed = new_pos_embed.permute(0, 2, 3, 1)
        new_pos_embed = new_pos_embed.view(tgt_size * tgt_size, dim)
        vision_pos_embed = torch.cat([cls_token, new_pos_embed], dim=0)
        vision_pos_embed = vision_pos_embed.view(1, tgt_size * tgt_size + 1, dim)
        return vision_pos_embed
    else:
        return abs_pos

@torch.jit.script
def quick_gelu(x):
    return x * torch.sigmoid(1.702 * x)



class CLIPVisionEmbeddings(nn.Module):
    def __init__(self, hidden_size=1024, image_size=224, patch_size=14, num_channels=3):
        super().__init__()
        self.embed_dim = hidden_size
        self.image_size = image_size
        self.patch_size = patch_size

        self.class_embedding = torch.nn.Parameter(torch.randn(self.embed_dim))

        self.patch_embedding = torch.nn.Conv2d(
            in_channels=num_channels,
            out_channels=self.embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            bias=False,
        )

        self.num_patches = (self.image_size // self.patch_size) ** 2
        self.num_positions = self.num_patches + 1
        self.position_embedding = torch.nn.Embedding(self.num_positions, self.embed_dim)
        self.register_buffer(
            "position_ids", torch.arange(self.num_positions).expand((1, -1))
        )

    def forward(self, pixel_values, patch_embeds):
        batch_size = pixel_values.shape[0]
        # patch_embeds = self.patch_embedding(
        #     pixel_values
        # )  # shape = [*, width, grid, grid]


        if patch_embeds is not None:
            patch_embeds = patch_embeds
            # print(patch_embeds.shape)
        else:
            patch_embeds = self.patch_embedding(pixel_values)  
            # print(111111)
        # shape = [*, width, grid, grid]
        # patch_embeds = patch_embeds.flatten(2).transpose(1, 2)

        patch_embeds = patch_embeds.flatten(2).transpose(1, 2)


        class_embeds = self.class_embedding.expand(batch_size, 1, -1)
        embeddings = torch.cat([class_embeds, patch_embeds], dim=1)

        # x = torch.cat([cls_token, x], dim=1)
        embeddings = embeddings + get_abs_pos(self.position_embedding(self.position_ids), embeddings.size(1))
        # embeddings = embeddings + self.position_embedding(self.position_ids)
        return embeddings


class NoTPFeedForward(nn.Module):
    def __init__(
            self,
            cfg,
            dim: int,
            hidden_dim: int,
    ):
        super().__init__()

        self.fc1 = torch.nn.Linear(dim, hidden_dim, bias=True)
        self.fc2 = torch.nn.Linear(hidden_dim, dim, bias=True)

    def forward(self, x):
        output = self.fc2(quick_gelu(self.fc1(x)))
        return output


# from optimus.flash_attn_interface import flash_attn_qkvpacked_func


# class NoTPAttention(nn.Module):
#     def __init__(self, cfg):
#         super().__init__()
#         self.num_heads = cfg.num_attention_heads
#         self.n_local_heads = cfg.num_attention_heads
#         self.head_dim = cfg.hidden_size // cfg.num_attention_heads
#         self.max_seq_len = cfg.seq_length
#         self.use_flash_attention = cfg.use_flash_attn

#         self.qkv_proj = torch.nn.Linear(cfg.hidden_size, cfg.hidden_size * 3, bias=True)
#         self.out_proj = torch.nn.Linear(cfg.hidden_size, cfg.hidden_size, bias=True)

#         # self.core_attention = CoreAttention(cfg, AttnType.self_attn)

#         self.attn_drop = cfg.attention_dropout

#     def forward(
#             self,
#             x: torch.Tensor,
#     ):
#         bsz, seqlen, _ = x.shape
#         xqkv = self.qkv_proj(x)
#         xqkv = xqkv.view(bsz, seqlen, 3, self.num_heads, self.head_dim)

#         if self.use_flash_attention:
#             output = flash_attn_qkvpacked_func(xqkv)
#             output = output.view(bsz, seqlen, -1)
#         else:
#             xq, xk, xv = torch.split(xqkv, 1, dim=2)
#             xq = xq.squeeze(2)
#             xk = xk.squeeze(2)
#             xv = xv.squeeze(2)
#             # xq, xk, xv = xqkv[:, :, 0, ...], xqkv[:, :, 1, ...], xqkv[:, :, 2, ...]

#             # （B, num_head, S, head_size)
#             xq = xq.permute(0, 2, 1, 3)
#             xk = xk.permute(0, 2, 1, 3)
#             xv = xv.permute(0, 2, 1, 3)

#             output = torch.nn.functional.scaled_dot_product_attention(xq, xk, xv, attn_mask=None)
#             utput = output.permute(0, 2, 1, 3).view(bsz, seqlen, -1)
#         output = self.out_proj(output)
#         return output


# from optimus.flash_attn_interface import flash_attn_qkvpacked_func


class NoTPAttention(torch.nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.num_heads = cfg.num_attention_heads
        self.n_local_heads = cfg.num_attention_heads
        self.head_dim = cfg.hidden_size // cfg.num_attention_heads
        self.max_seq_len = cfg.seq_length
        self.use_flash_attention = cfg.use_flash_attn

        self.qkv_proj = torch.nn.Linear(cfg.hidden_size, cfg.hidden_size * 3, bias=True)
        self.out_proj = torch.nn.Linear(cfg.hidden_size, cfg.hidden_size, bias=True)

        # self.core_attention = CoreAttention(cfg, AttnType.self_attn)

        self.attn_drop = cfg.attention_dropout

    def forward(
            self,
            x: torch.Tensor,
    ):
        bsz, seqlen, _ = x.shape
        xqkv = self.qkv_proj(x)
        xqkv = xqkv.view(bsz, seqlen, 3, self.num_heads, self.head_dim)

        if self.use_flash_attention:
            output = flash_attn_qkvpacked_func(xqkv)
            output = output.view(bsz, seqlen, -1)
            # xq, xk, xv = torch.split(xqkv, 1, dim=2)
            # xq = xq.squeeze(2)
            # xk = xk.squeeze(2)
            # xv = xv.squeeze(2)
            # # xq, xk, xv = xqkv[:, :, 0, ...], xqkv[:, :, 1, ...], xqkv[:, :, 2, ...]

            # # （B, num_head, S, head_size)
            # xq = xq.permute(0, 2, 1, 3)
            # xk = xk.permute(0, 2, 1, 3)
            # xv = xv.permute(0, 2, 1, 3)
            # # with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=False, enable_mem_efficient=False):
            # output = torch.nn.functional.scaled_dot_product_attention(xq, xk, xv, attn_mask=None)
            # output = output.permute(0, 2, 1, 3).reshape(bsz, seqlen, -1)
                # output = output.permute(0, 2, 1, 3).contiguous().view(bsz, seqlen, -1)
        else:
            # output = flash_attn_qkvpacked_func(xqkv)
            xq, xk, xv = torch.split(xqkv, 1, dim=2)
            xq = xq.squeeze(2)
            xk = xk.squeeze(2)
            xv = xv.squeeze(2)
            # xq, xk, xv = xqkv[:, :, 0, ...], xqkv[:, :, 1, ...], xqkv[:, :, 2, ...]

            # （B, num_head, S, head_size)
            xq = xq.permute(0, 2, 1, 3)
            xk = xk.permute(0, 2, 1, 3)
            xv = xv.permute(0, 2, 1, 3)
            # with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=False, enable_mem_efficient=False):
            output = torch.nn.functional.scaled_dot_product_attention(xq, xk, xv, attn_mask=None)
            output = output.permute(0, 2, 1, 3).reshape(bsz, seqlen, -1)
        output = self.out_proj(output)
        return output

class NoTPTransformerBlock(nn.Module):
    def __init__(self, cfg, layer_id: int, multiple_of=256):
        super().__init__()

        self.n_heads = cfg.num_attention_heads
        self.dim = cfg.hidden_size
        self.head_dim = cfg.hidden_size // cfg.num_attention_heads
        self.self_attn = NoTPAttention(cfg)
        self.mlp = NoTPFeedForward(
            cfg, dim=cfg.hidden_size, hidden_dim=cfg.ffn_hidden_size
        )
        self.layer_id = layer_id
        self.layer_norm1 = torch.nn.LayerNorm(
            cfg.hidden_size, eps=cfg.layernorm_epsilon
        )
        self.layer_norm2 = torch.nn.LayerNorm(
            cfg.hidden_size, eps=cfg.layernorm_epsilon
        )

    def forward(self, x: torch.Tensor):
        residual = self.self_attn.forward(self.layer_norm1(x))
        h = x + residual
        out = h + self.mlp.forward(self.layer_norm2(h))
        return out


class NoTPTransformer(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.cfg = cfg
        # self.recompute_list = self.cfg.get("recompute_list", [])
        self.num_layers = cfg.num_layers  # _get_num_layers(cfg)

        self.layers = torch.nn.ModuleList()
        for layer_id in range(self.num_layers):
            self.layers.append(
                NoTPTransformerBlock(
                    cfg,
                    layer_id + 1,
                )
            )

    def forward(
            self,
            hidden_states,
    ):

        for lid, layer in enumerate(self.layers):
            # if lid in self.recompute_list:
            #     def custom(layer_id):
            #         def custom_forward(*args, **kwargs):
            #             x_ = self.layers[layer_id](*args, **kwargs)
            #             return x_

            #         return custom_forward

            #     assert hidden_states.requires_grad == True, logger.warning(
            #         "When using recalculation, the input must have grad fn"
            #     )
            #     hidden_states = tensor_parallel.checkpoint(
            #         custom(lid),
            #         False,
            #         hidden_states.contiguous()
            #     )
            # else:
            hidden_states = layer(hidden_states)

        return hidden_states


# from megatron.core.tensor_parallel.layers import non_tensor_paralleled, local_dp_reduce, local_dp_scatter

class VitModel(nn.Module):
    def __init__(
            self,
            cfg,
            freeze_embed=False,
            freeze_pre_norm=False
    ) -> None:
        super().__init__()

        self.embeddings = CLIPVisionEmbeddings(hidden_size=cfg.hidden_size, image_size=cfg.image_size, patch_size=cfg.patch_size)

        if freeze_embed:
            for name, param in self.embeddings.named_parameters():
                param.requires_grad = False

        self.transformer = NoTPTransformer(cfg=cfg)

        if cfg.get("fp32norm", False):
            logger.info("Load fp32 layernorm for ViT.")
            self.pre_layrnorm = LayerNormfp32(
                cfg.hidden_size,
                eps=cfg.get("pre_layernorm_epsilon", 1e-5),
            )
        else:
            self.pre_layrnorm = torch.nn.LayerNorm(
                cfg.hidden_size,
                eps=cfg.get("pre_layernorm_epsilon", 1e-5),
            )

        # self.pre_layrnorm = RMSNorm(
        #     cfg.hidden_size,
        #     eps=cfg.get("pre_layernorm_epsilon", 1e-5),
        #     sequence_parallel=False,
        #     use_fp32=True,
        #     use_optimus=True,
        # )

        if freeze_pre_norm:
            for name, param in self.pre_layrnorm.named_parameters():
                param.requires_grad = False

        for p in self.parameters():
            p.micro_dp = True

    def set_input_tensor(self, input_tensor):
        if not isinstance(input_tensor, list):
            input_tensor = [input_tensor]
        self.transformer.set_input_tensor(input_tensor[0])

    def __str__(self) -> str:
        return "open_clip"

    def forward(
            self,
            x,
            patch_embeds
    ):
        x = self.embeddings(x, patch_embeds)
        hidden_states = self.pre_layrnorm(x)

        # hidden_states, dis = local_dp_scatter(hidden_states)
        output = self.transformer(hidden_states)

        # output = local_dp_reduce(output, dis)

        return output


vit_model_cfg = adict(
    num_layers=24,
    hidden_size=1024,
    num_heads = 16,
    num_attention_heads=16,
    ffn_hidden_size=4096,
    seq_length=256,
    max_position_embeddings=256,
    use_flash_attn=False,
    understand_projector_stride=2,
    hidden_dropout = 0.0,
    attention_dropout = 0.0,
    no_persist_layer_norm = False,
    layernorm_epsilon = 1e-5,
    pre_layernorm_epsilon = 1e-5,
    image_size = 224,
    patch_size = 14,
    recompute_list = []
)

def build_clip_l():
    return VitModel(
        cfg=vit_model_cfg,
        freeze_embed=False,
        freeze_pre_norm=False,
    )


if __name__ == '__main__':

    
    from mmgpt.model.vision_encoder.sam_b import build_sam_vit_b



    vit_model_cfg = adict(
        num_layers=24,
        hidden_size=1024,
        num_attention_heads=16,
        ffn_hidden_size=4096,
        seq_length=256,
        max_position_embeddings=256,
        use_flash_attn=False,
        understand_projector_stride=2,
        hidden_dropout = 0.0,
        attention_dropout = 0.0,
        no_persist_layer_norm = False,
        layernorm_epsilon = 1e-5,
        pre_layernorm_epsilon = 1e-5,
        image_size = 224,
        patch_size = 14,
        recompute_list = []
    )

    sam_model = build_sam_vit_b()


    vision_model = VitModel(
        cfg=vit_model_cfg,
        freeze_embed=False,
        freeze_pre_norm=False,
    )

    # model = VitModel(1344)
    # x = torch.zeros(2, 3, 224, 224)
    x = torch.zeros(2, 3, 1024, 1024)

    
    with torch.no_grad():
        # y = vision_model(x)
        patch_embed = sam_model(x)
        print(patch_embed.shape)
        y = vision_model(x, patch_embed)
        print(y.shape)

        image_feature = torch.add(y[:, 1:], patch_embed.flatten(2).permute(0, 2, 1))

        print(image_feature.shape)
