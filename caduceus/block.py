# Copyright (c) 2024, Tri Dao, Albert Gu.
from typing import Optional

import torch
from torch import nn, Tensor

from mamba_ssm.ops.triton.layer_norm import RMSNorm, layer_norm_fn
import torch.nn.functional as F


class GatedDilatedConvWithMLP(nn.Module):
    def __init__(
        self, hidden_size, layer_index, max_dilation=64, dilation_base=4, dropout=0.2,
        layers_per_module=8, conv_layers_per_module=5
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.layer_index = layer_index
        self.dilation_base = dilation_base
        self.layers_per_module = layers_per_module
        self.conv_layers_per_module = conv_layers_per_module

        # 计算在模块内的相对位置
        layer_in_module = layer_index % layers_per_module
        
        # 计算膨胀率：基于模块内的位置
        if layer_in_module < 1:
            self.dilation = 1
        else:
            self.dilation = min(
                dilation_base ** (layer_in_module - 1), max_dilation
            )
        self.dilation = int(self.dilation)

        # 计算 padding，确保是整数
        padding = (9 - 1) // 2 * self.dilation

        # 定义两个膨胀卷积
        self.conv_A = nn.Conv1d(
            hidden_size,
            hidden_size,
            kernel_size=9,
            padding=padding,
            dilation=self.dilation,
            groups=hidden_size,
        )

        self.conv_B = nn.Conv1d(
            hidden_size,
            hidden_size,
            kernel_size=9,
            padding=padding,
            dilation=self.dilation,
            groups=hidden_size,
        )

        # MLP
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.LayerNorm(hidden_size),
        )
        # dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, L, D = x.shape
        x_conv = x.permute(0, 2, 1)  # (B, D, L)
        
        # 计算在模块内的相对位置
        layer_in_module = self.layer_index % self.layers_per_module
        
        if layer_in_module > self.conv_layers_per_module - 1:
            h = self.dropout(x_conv)
        # 计算 h 和 g
        h = F.gelu(self.conv_A(x_conv))  # (B, D, L)
        g = torch.sigmoid(self.conv_B(x_conv))  # (B, D, L)

        # 门控 + MLP
        x_conv = x_conv + h * g
        x = x_conv.permute(0, 2, 1)  # (B, L, D)
        x = self.mlp(x) + x

        return x


class Block(nn.Module):
    features = []

    def __init__(
        self,
        dim,
        mixer_cls,
        mlp_cls,
        layer_idx=None,
        attn_layer_idx=None,  # 新增参数，指定注意力层索引
        MSC_layer_idx=None,
        dilation_base=4,
        dropout=0.0,
        layers_per_module=8,      # 新增参数
        conv_layers_per_module=5, # 新增参数
        norm_cls=nn.LayerNorm,
        fused_add_norm=False,
        residual_in_fp32=False,
    ):
        """
        Simple block wrapping a mixer class with LayerNorm/RMSNorm and residual connection"

        This Block has a slightly different structure compared to a regular
        prenorm Transformer block.
        The standard block is: LN -> MHA/MLP -> Add.
        [Ref: https://arxiv.org/abs/2002.04745]
        Here we have: Add -> LN -> Mixer, returning both
        the hidden_states (output of the mixer) and the residual.
        This is purely for performance reasons, as we can fuse add and LayerNorm.
        The residual needs to be provided (except for the very first block).
        """
        super().__init__()
        self.residual_in_fp32 = residual_in_fp32
        self.fused_add_norm = fused_add_norm
        self.norm = norm_cls(dim)
        self.layer_idx = layer_idx
        self.MSC_layer_idx = MSC_layer_idx
        self.attn_layer_idx = attn_layer_idx
        # self.conv_block = None
        self.mixer = mixer_cls(dim)
        if self.layer_idx in MSC_layer_idx:
            self.convnorm = norm_cls(dim)
            self.conv_block = GatedDilatedConvWithMLP(
                dim, self.layer_idx, 
                dilation_base=dilation_base, 
                dropout=dropout,
                layers_per_module=layers_per_module,
                conv_layers_per_module=conv_layers_per_module
            )
        else:
            self.conv_block = None
        if mlp_cls is not nn.Identity and self.conv_block is None:
            self.norm2 = norm_cls(dim)
            self.mlp = mlp_cls(dim)
        else:
            self.mlp = None
        if self.fused_add_norm:
            assert RMSNorm is not None, "RMSNorm import fails"
            assert isinstance(self.norm, (nn.LayerNorm, RMSNorm)), (
                "Only LayerNorm and RMSNorm are supported for fused_add_norm"
            )

    def forward(
        self,
        hidden_states: Tensor,
        residual: Optional[Tensor] = None,
        inference_params=None,
        **mixer_kwargs,
    ):
        r"""Pass the input through the encoder layer.

        Args:
            hidden_states: the sequence to the encoder layer (required).
            residual: hidden_states = Mixer(LN(residual))
        """
        if not self.fused_add_norm:
            residual = (
                (hidden_states + residual) if residual is not None else hidden_states
            )
            hidden_states = self.norm(residual.to(dtype=self.norm.weight.dtype))
            if self.residual_in_fp32:
                residual = residual.to(torch.float32)
        else:
            hidden_states, residual = layer_norm_fn(
                hidden_states,
                self.norm.weight,
                self.norm.bias,
                residual=residual,
                prenorm=True,
                residual_in_fp32=self.residual_in_fp32,
                eps=self.norm.eps,
                is_rms_norm=isinstance(self.norm, RMSNorm),
            )
        hidden_states = self.mixer(
            hidden_states, inference_params=inference_params, **mixer_kwargs
        )
        if self.conv_block is not None:
            if not self.fused_add_norm:
                residual = hidden_states + residual
                hidden_states = self.convnorm(
                    residual.to(dtype=self.convnorm.weight.dtype)
                )
                if self.residual_in_fp32:
                    residual = residual.to(torch.float32)
            else:
                hidden_states, residual = layer_norm_fn(
                    hidden_states,
                    self.convnorm.weight,
                    self.convnorm.bias,
                    residual=residual,
                    prenorm=True,
                    residual_in_fp32=self.residual_in_fp32,
                    eps=self.convnorm.eps,
                    is_rms_norm=isinstance(self.convnorm, RMSNorm),
                )
            hidden_states = self.conv_block(hidden_states)
        if self.mlp is not None:
            if not self.fused_add_norm:
                residual = hidden_states + residual
                hidden_states = self.norm2(residual.to(dtype=self.norm2.weight.dtype))
                if self.residual_in_fp32:
                    residual = residual.to(torch.float32)
            else:
                hidden_states, residual = layer_norm_fn(
                    hidden_states,
                    self.norm2.weight,
                    self.norm2.bias,
                    residual=residual,
                    prenorm=True,
                    residual_in_fp32=self.residual_in_fp32,
                    eps=self.norm2.eps,
                    is_rms_norm=isinstance(self.norm2, RMSNorm),
                )
            hidden_states = self.mlp(hidden_states)

        return hidden_states, residual

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return self.mixer.allocate_inference_cache(
            batch_size, max_seqlen, dtype=dtype, **kwargs
        )
