# Copyright (c) 2024, Tri Dao, Albert Gu.

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

try:
    from flash_attn import (
        flash_attn_kvpacked_func,
        flash_attn_qkvpacked_func,
        flash_attn_varlen_kvpacked_func,
        flash_attn_varlen_qkvpacked_func,
        flash_attn_with_kvcache,
    )
except ImportError:
    flash_attn_varlen_qkvpacked_func, flash_attn_varlen_kvpacked_func = None, None
    flash_attn_qkvpacked_func, flash_attn_kvpacked_func = None, None
    flash_attn_with_kvcache = None

try:
    from flash_attn.layers.rotary import RotaryEmbedding
except ImportError:
    RotaryEmbedding = None

try:
    from causal_conv1d import causal_conv1d_fn, causal_conv1d_update
except ImportError:
    causal_conv1d_fn, causal_conv1d_update = None, None

# Import Fourier Position Embedding
from .fourier_position_embedding import FourierEmbedding, RotaryEmbedding


def get_alibi_slopes(nheads):
    def get_slopes_power_of_2(nheads):
        start = 2 ** (-(2 ** -(math.log2(nheads) - 3)))
        ratio = start
        return [start * ratio**i for i in range(nheads)]

    if math.log2(nheads).is_integer():
        return get_slopes_power_of_2(nheads)
    else:
        closest_power_of_2 = 2 ** math.floor(math.log2(nheads))
        return (
            get_slopes_power_of_2(closest_power_of_2)
            + get_alibi_slopes(2 * closest_power_of_2)[0::2][
                : nheads - closest_power_of_2
            ]
        )


def _update_kv_cache(kv, inference_params, layer_idx):
    """kv: (batch_size, seqlen, 2, nheads, head_dim) or (batch_size, 1, 2, nheads, head_dim)"""
    # Pre-allocate memory for key-values for inference.
    num_heads, head_dim = kv.shape[-2:]
    assert layer_idx in inference_params.key_value_memory_dict
    kv_cache, _ = inference_params.key_value_memory_dict[layer_idx]
    # Adjust key and value for inference
    batch_start = inference_params.batch_size_offset
    batch_end = batch_start + kv.shape[0]
    sequence_start = inference_params.seqlen_offset
    sequence_end = sequence_start + kv.shape[1]
    assert batch_end <= kv_cache.shape[0]
    assert sequence_end <= kv_cache.shape[1]
    assert kv_cache is not None
    kv_cache[batch_start:batch_end, sequence_start:sequence_end, ...] = kv
    return kv_cache[batch_start:batch_end, :sequence_end, ...]


class MHA(nn.Module):
    """Multi-head self-attention and cross-attention"""

    LOAD_BALANCING_LOSSES = []

    def __init__(
        self,
        embed_dim,
        num_heads,
        num_heads_kv=None,
        head_dim=None,  # If None, use embed_dim // num_heads
        mlp_dim=0,
        dropout=0.0,
        qkv_proj_bias=True,
        out_proj_bias=True,
        softmax_scale=None,
        causal=False,  # 因果卷积
        layer_idx=None,
        d_conv=0,
        rotary_emb_dim=0,
        rotary_emb_base=10000.0,
        rotary_emb_interleaved=False,
        # Fourier Position Embedding parameters
        use_fourier_pos_emb=False,
        fourier_max_seq_len=32768,
        fourier_dim=None,
        fourier_init="eye_xavier_norm",
        fourier_init_norm_gain=0.3,
        fourier_separate_basis=True,
        fourier_separate_head=True,
        fourier_learnable=True,
        fourier_norm=False,
        fourier_ignore_zero=True,
        window_size=(-1, -1),
        deterministic=False,
        use_flash_attn=False,
        use_alibi=False,
        use_moh=False,
        device=None,
        dtype=None,
    ) -> None:
        """
        num_heads_kv: can be used to toggle MQA / GQA. If None, use num_heads.
        return_residual: whether to return the input x along with the output. This is for
            performance reason: for post-norm architecture, returning the input allows us
            to fuse the backward of nn.Linear with the residual connection.
        """
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.embed_dim = embed_dim
        self.layer_idx = layer_idx
        self.d_conv = d_conv
        self.rotary_emb_dim = rotary_emb_dim
        self.use_fourier_pos_emb = use_fourier_pos_emb
        self.softmax_scale = softmax_scale
        self.causal = causal
        self.window_size = window_size
        self.deterministic = deterministic
        self.drop = nn.Dropout(dropout)
        self.use_flash_attn = use_flash_attn
        self.routed_head = 0
        self.shared_head = 0
        self.use_moh = use_moh
        current_device = torch.cuda.current_device()

        if use_alibi:
            assert use_flash_attn, "ALiBi code path requires flash_attn"
            self.alibi_slopes = torch.tensor(
                get_alibi_slopes(num_heads), **factory_kwargs
            )
            self.alibi_slopes = self.alibi_slopes.to(current_device)
        else:
            self.alibi_slopes = None
        if window_size != (-1, -1):
            assert use_flash_attn, (
                "Local (sliding window) attention code path requires flash_attn"
            )

        self.num_heads = num_heads
        self.num_heads_kv = num_heads_kv if num_heads_kv is not None else num_heads
        assert self.num_heads % self.num_heads_kv == 0, (
            "num_heads must be divisible by num_heads_kv"
        )
        if head_dim is None:
            assert self.embed_dim % num_heads == 0, (
                "embed_dim must be divisible by num_heads"
            )
        self.head_dim = (
            head_dim if head_dim is not None else self.embed_dim // num_heads
        )
        self.mlp_dim = math.ceil(mlp_dim / 256) * 256
        qkv_dim = self.head_dim * (self.num_heads + 2 * self.num_heads_kv)
        out_dim = self.head_dim * self.num_heads

        self.pos_emb = None
        if self.use_fourier_pos_emb:
            # Create a config object for FourierEmbedding
            class EmbConfig:
                def __init__(self):
                    self.d_model = embed_dim
                    self.n_heads = num_heads
                    self.attn_cfg = {'num_heads': num_heads, 'rotary_emb_base': rotary_emb_base}
                    self.fourier_max_seq_len = fourier_max_seq_len
                    self.fourier_dim = fourier_dim
                    self.fourier_init = fourier_init
                    self.fourier_init_norm_gain = fourier_init_norm_gain
                    self.fourier_learnable = fourier_learnable
                    self.fourier_separate_basis = fourier_separate_basis
                    self.fourier_separate_head = fourier_separate_head
                    self.fourier_norm = fourier_norm
                    self.fourier_ignore_zero = fourier_ignore_zero
                    self.rope_theta = rotary_emb_base
            
            emb_config = EmbConfig()
            self.pos_emb = FourierEmbedding(emb_config, device=device, dtype=dtype)
        elif self.rotary_emb_dim > 0:
            assert RotaryEmbedding is not None, (
                "rotary requires flash_attn to be installed"
            )
            # Note: Using the flash_attn RotaryEmbedding, not our custom one
            from flash_attn.layers.rotary import RotaryEmbedding as FlashRotaryEmbedding
            self.pos_emb = FlashRotaryEmbedding(
                self.rotary_emb_dim,
                base=rotary_emb_base,
                interleaved=rotary_emb_interleaved,
                device=device,
            )
        
        # 为了兼容原始 RoPE 实现，保持 rotary_emb 属性
        if self.rotary_emb_dim > 0 and not self.use_fourier_pos_emb:
            self.rotary_emb = self.pos_emb

        self.in_proj = nn.Linear(
            embed_dim, qkv_dim + self.mlp_dim, bias=qkv_proj_bias, **factory_kwargs
        )
        if self.d_conv > 0:
            if not self.causal:
                assert self.d_conv % 2 == 1, (
                    "卷积核大小必须为奇数，当不使用因果卷积时。"
                )
                padding = (self.d_conv - 1) // 2
            else:
                padding = self.d_conv - 1

            self.conv1d = nn.Conv1d(
                qkv_dim,
                qkv_dim,
                kernel_size=self.d_conv,
                padding=padding,  # 动态设置 padding
                groups=qkv_dim,
                **factory_kwargs,
            )

        self.out_proj = nn.Linear(
            out_dim + self.mlp_dim // 2, embed_dim, bias=out_proj_bias, **factory_kwargs
        )
        # 混合头注意力加入。
        if self.use_moh:
            #################MoH#################
            self.routed_head = int(self.num_heads * (0.75 - 0.5))
            self.shared_head = int(self.num_heads * (0.5))
            #################MoH#################

        if self.routed_head > 0:
            self.wg = torch.nn.Linear(
                embed_dim, num_heads - self.shared_head, bias=False, **factory_kwargs
            )
            if self.shared_head > 0:
                self.wg_0 = torch.nn.Linear(embed_dim, 2, bias=False, **factory_kwargs)

        if self.shared_head > 1:
            self.wg_1 = torch.nn.Linear(
                embed_dim, self.shared_head, bias=False, **factory_kwargs
            )

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None):
        dtype = self.out_proj.weight.dtype if dtype is None else dtype
        device = self.out_proj.weight.device
        if self.d_conv > 0:
            conv_state = torch.zeros(
                batch_size,
                self.conv1d.weight.shape[0],
                self.d_conv,
                device=device,
                dtype=dtype,
            )
        else:
            conv_state = None
        kv_cache = torch.empty(
            batch_size,
            max_seqlen,
            2,
            self.num_heads_kv,
            self.head_dim,
            dtype=dtype,
            device=device,
        )
        return kv_cache, conv_state

    def _update_kv_cache(self, kv, inference_params):
        """kv: (batch_size, seqlen, 2, nheads, head_dim) or (batch_size, 1, 2, nheads, head_dim)"""
        assert self.layer_idx is not None, (
            "Generation requires layer_idx in the constructor"
        )
        return _update_kv_cache(kv, inference_params, self.layer_idx)

    def _apply_rotary_update_kvcache_attention(self, q, kv, inference_params):
        """
        Fast path that combine 3 steps: apply rotary to Q and K, update kv cache, and apply attention.
        q: (batch_size, seqlen_q, nheads, head_dim)
        kv: (batch_size, seqlen_k, 2, nheads_kv, head_dim)
        """
        assert inference_params is not None and inference_params.seqlen_offset > 0
        if self.rotary_emb_dim > 0:
            self.rotary_emb._update_cos_sin_cache(
                inference_params.max_seqlen, device=q.device, dtype=q.dtype
            )
            rotary_cos, rotary_sin = (
                self.rotary_emb._cos_cached,
                self.rotary_emb._sin_cached,
            )
        else:
            rotary_cos, rotary_sin = None, None
        batch = q.shape[0]
        kv_cache, _ = inference_params.key_value_memory_dict[self.layer_idx]
        kv_cache = kv_cache[:batch]
        cache_seqlens = (
            inference_params.lengths_per_sample[:batch]
            if inference_params.lengths_per_sample is not None
            else inference_params.seqlen_offset
        )
        assert flash_attn_with_kvcache is not None, "flash_attn must be installed"
        context = flash_attn_with_kvcache(
            q,
            kv_cache[:, :, 0],
            kv_cache[:, :, 1],
            kv[:, :, 0],
            kv[:, :, 1],
            rotary_cos=rotary_cos,
            rotary_sin=rotary_sin,
            cache_seqlens=cache_seqlens,
            softmax_scale=self.softmax_scale,
            causal=self.causal,
            rotary_interleaved=self.rotary_emb.interleaved
            if self.rotary_emb_dim > 0
            else False,
        )
        return context

    def _update_kvcache_attention(self, q, kv, inference_params):
        """Write kv to inference_params, then do attention"""
        if inference_params.seqlen_offset == 0 or flash_attn_with_kvcache is None:
            # TODO: this only uses seqlen_offset and not lengths_per_sample.
            kv = self._update_kv_cache(kv, inference_params)
            k, v = kv.unbind(dim=-3)
            k = torch.repeat_interleave(
                k, dim=2, repeats=self.num_heads // self.num_heads_kv
            )
            v = torch.repeat_interleave(
                v, dim=2, repeats=self.num_heads // self.num_heads_kv
            )
            return F.scaled_dot_product_attention(
                q.transpose(1, 2),
                k.transpose(1, 2),
                v.transpose(1, 2),
                is_causal=self.causal,
                scale=self.softmax_scale,
            ).transpose(1, 2)
        else:
            batch = q.shape[0]
            kv_cache, _ = inference_params.key_value_memory_dict[self.layer_idx]
            kv_cache = kv_cache[:batch]
            cache_seqlens = (
                inference_params.lengths_per_sample[:batch]
                if inference_params.lengths_per_sample is not None
                else inference_params.seqlen_offset
            )
            return flash_attn_with_kvcache(
                q,
                kv_cache[:, :, 0],
                kv_cache[:, :, 1],
                kv[:, :, 0],
                kv[:, :, 1],
                cache_seqlens=cache_seqlens,
                softmax_scale=self.softmax_scale,
                causal=self.causal,
            )

    def forward(self, x, inference_params=None):
        """
        Arguments:
            x: (batch, seqlen, hidden_dim) (where hidden_dim = num heads * head dim) if
                cu_seqlens is None and max_seqlen is None, else (total, hidden_dim) where total
                is the is the sum of the sequence lengths in the batch.
            inference_params: for generation. Adapted from Megatron-LM (and Apex)
            https://github.com/NVIDIA/apex/blob/3ff1a10f72ec07067c4e44759442329804ac5162/apex/transformer/testing/standalone_transformer_lm.py#L470
        """
        if (
            inference_params is not None
            and self.layer_idx not in inference_params.key_value_memory_dict
        ):
            inference_params.key_value_memory_dict[self.layer_idx] = (
                self.allocate_inference_cache(
                    x.shape[0], inference_params.max_seqlen, dtype=x.dtype
                )
            )
        
        # 添加与原始代码兼容的 seqlen_offset 和 rotary_max_seqlen
        seqlen_offset = (
            0
            if inference_params is None
            else (
                inference_params.lengths_per_sample
                if inference_params.lengths_per_sample is not None
                else inference_params.seqlen_offset
            )
        )
        rotary_max_seqlen = (
            inference_params.max_seqlen if inference_params is not None else None
        )
        
        # 添加混合头注意力。
        if self.routed_head > 0:
            B, N, C = x.shape
            _x = x.reshape(B * N, C)
            logits = self.wg(_x)
            gates = F.softmax(logits, dim=1)

            num_tokens, num_experts = gates.shape
            _, indices = torch.topk(gates, k=self.routed_head, dim=1)
            mask = F.one_hot(indices, num_classes=num_experts).sum(dim=1)

            if self.training:
                me = gates.mean(dim=0)
                ce = mask.float().mean(dim=0)
                l_aux = torch.mean(me * ce) * num_experts * num_experts

                MHA.LOAD_BALANCING_LOSSES.append(l_aux)

            routed_head_gates = gates * mask
            denom_s = torch.sum(routed_head_gates, dim=1, keepdim=True)
            denom_s = torch.clamp(denom_s, min=torch.finfo(denom_s.dtype).eps)
            routed_head_gates /= denom_s
            routed_head_gates = routed_head_gates.reshape(B, N, -1) * self.routed_head

        qkv = self.in_proj(x)
        if self.mlp_dim > 0:
            qkv, x_mlp = qkv.split([qkv.shape[-1] - self.mlp_dim, self.mlp_dim], dim=-1)
            x_mlp_up, x_mlp_gate = x_mlp.chunk(2, dim=-1)
            x_mlp = x_mlp_up * F.silu(x_mlp_gate)
        # 卷积过程
        if self.d_conv > 0:
            # The inference code for conv1d is pretty messy, should clean it up
            if inference_params is None or inference_params.seqlen_offset == 0:
                if self.causal:
                    # 因果卷积
                    if causal_conv1d_fn is None:
                        qkv = rearrange(
                            self.conv1d(rearrange(qkv, "b s d -> b d s"))[
                                ..., : -(self.d_conv - 1)
                            ],
                            "b d s -> b s d",
                        ).contiguous()
                    else:
                        qkv = causal_conv1d_fn(
                            qkv.transpose(1, 2),
                            rearrange(self.conv1d.weight, "d 1 w -> d w"),
                            self.conv1d.bias,
                        ).transpose(1, 2)
                else:
                    # 正常的一维深度卷积
                    qkv = rearrange(
                        self.conv1d(rearrange(qkv, "b s d -> b d s")),
                        "b d s -> b s d",
                    ).contiguous()

                if inference_params is not None:
                    _, conv_state = inference_params.key_value_memory_dict[
                        self.layer_idx
                    ]
                    # If we just take qkv[:, :, -self.d_conv :], it will error if seqlen < self.d_conv
                    # Instead F.pad will pad with zeros if seqlen < self.d_conv, and truncate otherwise.
                    qkv_t = rearrange(qkv, "b l d -> b d l")
                    conv_state.copy_(
                        F.pad(qkv_t, (self.d_conv - qkv_t.shape[-1], 0))
                    )  # Update state (B D W)
            else:
                _, conv_state = inference_params.key_value_memory_dict[self.layer_idx]
                assert qkv.shape[1] == 1, (
                    "Only support decoding with 1 token at a time for now"
                )
                qkv = qkv.squeeze(1)
                # Conv step
                if self.causal:
                    if causal_conv1d_update is None:
                        conv_state.copy_(
                            torch.roll(conv_state, shifts=-1, dims=-1)
                        )  # Update state (B D W)
                        conv_state[:, :, -1] = qkv
                        qkv = torch.sum(
                            conv_state * rearrange(self.conv1d.weight, "d 1 w -> d w"),
                            dim=-1,
                        )  # (B D)
                        if self.conv1d.bias is not None:
                            qkv = qkv + self.conv1d.bias
                    else:
                        qkv = causal_conv1d_update(
                            qkv,
                            conv_state,
                            rearrange(self.conv1d.weight, "d 1 w -> d w"),
                            self.conv1d.bias,
                        )
                else:
                    # 正常卷积更新状态
                    qkv = torch.sum(
                        conv_state * rearrange(self.conv1d.weight, "d 1 w -> d w"),
                        dim=-1,
                    )  # (B D)
                    if self.conv1d.bias is not None:
                        qkv = qkv + self.conv1d.bias
                qkv = qkv.unsqueeze(1)

        q, kv = qkv.split(
            [self.num_heads * self.head_dim, self.num_heads_kv * 2 * self.head_dim],
            dim=-1,
        )
        q = rearrange(q, "... (h d) -> ... h d", d=self.head_dim)
        kv = rearrange(kv, "... (two hkv d) -> ... two hkv d", two=2, d=self.head_dim)
        
        if (
            inference_params is None
            or inference_params.seqlen_offset == 0
            or (self.rotary_emb_dim == 0 or self.rotary_emb_dim % 16 != 0)
        ):
            # Apply position embedding based on the type
            if self.use_fourier_pos_emb and self.pos_emb is not None:
                # Apply Fourier Position Embedding
                # Convert to the format expected by FourierEmbedding
                # q shape: (batch, seqlen, num_heads, head_dim)
                # Need: (batch, num_heads, seqlen, head_dim)
                q_transposed = q.transpose(1, 2)
                kv_unbound = kv.unbind(dim=2)
                k_transposed = kv_unbound[0].transpose(1, 2)
                
                # Apply FoPE with proper sequence length
                seq_len = q.size(1)  # seqlen dimension
                q_transformed = self.pos_emb(q_transposed, seq_len)
                k_transformed = self.pos_emb(k_transposed, seq_len)

                # Convert back to original format
                q = q_transformed.transpose(1, 2)
                k = k_transformed.transpose(1, 2)
                kv = torch.stack([k, kv_unbound[1]], dim=2)
            elif self.rotary_emb_dim > 0 and not self.use_fourier_pos_emb:
                # Apply Rotary Position Embedding (original implementation)
                q, kv = self.rotary_emb(
                    q, kv, seqlen_offset=seqlen_offset, max_seqlen=rotary_max_seqlen
                )

            if inference_params is None:
                k, v = kv.unbind(dim=-3)
                k = torch.repeat_interleave(
                    k, dim=2, repeats=self.num_heads // self.num_heads_kv
                )
                v = torch.repeat_interleave(
                    v, dim=2, repeats=self.num_heads // self.num_heads_kv
                )
                if self.use_flash_attn:
                    qkv_combined = torch.stack([q, k, v], dim=-3)
                    context = flash_attn_qkvpacked_func(
                        qkv_combined,
                        self.drop.p if self.training else 0.0,
                        softmax_scale=self.softmax_scale,
                        causal=self.causal,
                        alibi_slopes=self.alibi_slopes,
                        window_size=self.window_size,
                        deterministic=self.deterministic,
                    )
                else:
                    context = F.scaled_dot_product_attention(
                        q.transpose(1, 2),
                        k.transpose(1, 2),
                        v.transpose(1, 2),
                        dropout_p=self.drop.p if self.training else 0.0,
                        is_causal=self.causal,
                        scale=self.softmax_scale,
                    ).transpose(1, 2)

            else:
                context = self._update_kvcache_attention(q, kv, inference_params)
        else:
            context = self._apply_rotary_update_kvcache_attention(
                q, kv, inference_params
            )
        # 添加混合头注意力。
        if self.routed_head > 0:
            if self.shared_head > 0:
                shared_head_weight = self.wg_1(_x)
                shared_head_gates = (
                    F.softmax(shared_head_weight, dim=1).reshape(B, N, -1)
                    * self.shared_head
                )

                weight_0 = self.wg_0(_x)
                weight_0 = F.softmax(weight_0, dim=1).reshape(B, N, 2) * 2

                shared_head_gates = torch.einsum(
                    "bn,bne->bne", weight_0[:, :, 0], shared_head_gates
                )
                routed_head_gates = torch.einsum(
                    "bn,bne->bne", weight_0[:, :, 1], routed_head_gates
                )

                masked_gates = torch.cat([shared_head_gates, routed_head_gates], dim=2)
            else:
                masked_gates = routed_head_gates

            context = torch.einsum("bne,bned->bned", masked_gates, context)
            context = rearrange(context, "... h d -> ... (h d)")
        else:
            context = rearrange(context, "... h d -> ... (h d)")
        if self.mlp_dim > 0:
            context = torch.cat([context, x_mlp], dim=-1)
        out = self.out_proj(context)
        return out
