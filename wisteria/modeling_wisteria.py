"""Wisteria model for Hugging Face."""

import copy
import inspect
import math
from functools import partial
from typing import Optional, Tuple, Union

import torch
from mamba_ssm.modules.mamba2 import Mamba2
from mamba_ssm.modules.mamba_simple import Mamba
from .mha import MHA
from mamba_ssm.modules.mlp import GatedMLP

from .block import Block
from torch import nn
from torch.nn import functional as F
from transformers import PreTrainedModel
from transformers.modeling_outputs import (
    BaseModelOutputWithNoAttention,
    MaskedLMOutput,
    SequenceClassifierOutput,
)

try:
    from mamba_ssm.ops.triton.layernorm import (  # Legacy mambav1 file structure
        RMSNorm,
        layer_norm_fn,
        rms_norm_fn,
    )
except ImportError:
    try:
        from mamba_ssm.ops.triton.layer_norm import (  # mambav2 file structure
            RMSNorm,
            layer_norm_fn,
            rms_norm_fn,
        )
    except ImportError:
        RMSNorm, layer_norm_fn, rms_norm_fn = None, None, None

from .configuration_wisteria import WisteriaConfig
from .modeling_rcps import RCPSAddNormWrapper, RCPSEmbedding, RCPSLMHead, RCPSMambaBlock


def create_block(
    d_model,
    d_intermediate,  # 修改1：增加d_intermediate参数，用于指定是否在块中增加该维度的mlp。
    MSC_layer_idx=None,  # 修改8：增加MSC_later_idx参数，用于指定哪些层使用多尺度卷积模块。
    dilation_base=4,  #   # 修改9：增加dilation_base参数，用于指定多尺度卷积模块的基础膨胀率。
    dropout=0.0,
    layers_per_module=8,      # 新增参数
    conv_layers_per_module=5, # 新增参数
    ssm_cfg=None,
    norm_epsilon=1e-5,
    rms_norm=False,
    residual_in_fp32=False,
    fused_add_norm=False,
    layer_idx=None,
    bidirectional=True,
    bidirectional_strategy="add",
    bidirectional_weight_tie=True,
    rcps=False,  # 增加rcps参数，用于指定是否使用RCPS模块，否则使用mamba block。
    device=None,
    dtype=None,
    attn_layer_idx=None,  # 修改2：增加attn_layer_idx参数，用于指定哪些层是注意力层。
    attn_cfg=None,  # 修改3：增加attn_cfg参数，用于指定注意力层的配置。
    # Fourier Position Embedding parameters
    use_fourier_pos_emb=False,
    fourier_max_seq_len=32768,
    fourier_dim=None,
    fourier_init="eye_xavier_norm",
    fourier_init_norm_gain=0.3,
    fourier_separate_basis=True,
    fourier_separate_head=True,
    fourier_learnable=False,
    fourier_norm=False,
    fourier_ignore_zero=True,
):
    """Create Wisteria block.

    Adapted from: https://github.com/state-spaces/mamba/blob/main/mamba_ssm/models/mixer_seq_simple.py
    """
    # 定义一些参数
    if ssm_cfg is None:
        ssm_cfg = {}
    if attn_layer_idx is None:
        attn_layer_idx = []
    if attn_cfg is None:
        attn_cfg = {}
    if MSC_layer_idx is None:
        MSC_layer_idx = []

    # 定义一些参数
    factory_kwargs = {"device": device, "dtype": dtype}
    bidirectional_kwargs = {
        "bidirectional": bidirectional,
        "bidirectional_strategy": bidirectional_strategy,
        "bidirectional_weight_tie": bidirectional_weight_tie,
    }

    # 如果当前层数不属于attn_layer_idx，则使用BiMambaWrapper，否则使用mha,表明当前层是注意力层。
    if layer_idx not in attn_layer_idx:
        mixer_cls = partial(
            BiMambaWrapper,
            layer_idx=layer_idx,
            **ssm_cfg,
            **bidirectional_kwargs,
            **factory_kwargs,
        )
    else:
        # 为注意力层添加傅里叶位置编码配置
        enhanced_attn_cfg = attn_cfg.copy() if attn_cfg else {}
        
        # 添加傅里叶位置编码配置
        enhanced_attn_cfg.update({
            'use_fourier_pos_emb': use_fourier_pos_emb,
            'fourier_max_seq_len': fourier_max_seq_len,
            'fourier_dim': fourier_dim,
            'fourier_init': fourier_init,
            'fourier_init_norm_gain': fourier_init_norm_gain,
            'fourier_separate_basis': fourier_separate_basis,
            'fourier_separate_head': fourier_separate_head,
            'fourier_learnable': fourier_learnable,
            'fourier_norm': fourier_norm,
            'fourier_ignore_zero': fourier_ignore_zero,
        })
        
        mixer_cls = partial(MHA, layer_idx=layer_idx, **enhanced_attn_cfg, **factory_kwargs)

    norm_cls = partial(
        nn.LayerNorm if not rms_norm else RMSNorm, eps=norm_epsilon, **factory_kwargs
    )
    # 定义块
    block_cls = RCPSMambaBlock if rcps else Block

    # 根据参数d_intermediate选择是否在块中增加mlp
    if d_intermediate == 0:
        mlp_cls = nn.Identity
    else:
        mlp_cls = partial(
            GatedMLP,
            hidden_features=d_intermediate,
            out_features=d_model,
            **factory_kwargs,
        )

    block = block_cls(
        d_model,
        mixer_cls,
        mlp_cls,
        layer_idx,
        attn_layer_idx,
        MSC_layer_idx,
        dilation_base,
        dropout,
        layers_per_module,
        conv_layers_per_module,
        norm_cls=norm_cls,
        fused_add_norm=fused_add_norm,
        residual_in_fp32=residual_in_fp32,
    )
    block.layer_idx = layer_idx
    return block


class BiMambaWrapper(nn.Module):
    """Thin wrapper around Mamba to support bi-directionality."""

    def __init__(
        self,
        d_model: int,
        bidirectional: bool = True,
        bidirectional_strategy: Optional[str] = "add",
        bidirectional_weight_tie: bool = True,
        **mamba_kwargs,
    ):
        super().__init__()
        if bidirectional and bidirectional_strategy is None:
            bidirectional_strategy = "add"  # Default strategy: `add`
        if bidirectional and bidirectional_strategy not in ["add", "ew_multiply"]:
            raise NotImplementedError(
                f"`{bidirectional_strategy}` strategy for bi-directionality is not implemented!"
            )
        self.bidirectional = bidirectional
        self.bidirectional_strategy = bidirectional_strategy

        # 判断mamba2是否使用并行化

        self.mamba_fwd = Mamba(d_model=d_model, **mamba_kwargs)
        if bidirectional:
            self.mamba_rev = Mamba(d_model=d_model, **mamba_kwargs)
            if (
                bidirectional_weight_tie
            ):  # Tie in and out projections (where most of param count lies)
                self.mamba_rev.in_proj.weight = self.mamba_fwd.in_proj.weight
                self.mamba_rev.in_proj.bias = self.mamba_fwd.in_proj.bias
                self.mamba_rev.out_proj.weight = self.mamba_fwd.out_proj.weight
                self.mamba_rev.out_proj.bias = self.mamba_fwd.out_proj.bias
        else:
            self.mamba_rev = None

    def forward(self, hidden_states, inference_params=None):
        """Bidirectional-enabled forward pass

        hidden_states: (B, L, D)
        Returns: same shape as hidden_states
        """
        out = self.mamba_fwd(hidden_states, inference_params=inference_params)

        # 将原始的mamba2层变为双向mamba2层
        if self.bidirectional:
            out_rev = self.mamba_rev(
                hidden_states.flip(
                    dims=(1,)
                ),  # Flip along the sequence length dimension
                inference_params=inference_params,
            ).flip(dims=(1,))  # Flip back for combining with forward hidden states
            if self.bidirectional_strategy == "add":
                out = out + out_rev
            elif self.bidirectional_strategy == "ew_multiply":
                out = out * out_rev
            else:
                raise NotImplementedError(
                    f"`{self.bidirectional_strategy}` for bi-directionality not implemented!"
                )
        return out


class WisteriaEmbeddings(nn.Module):
    def __init__(
        self,
        config: WisteriaConfig,
        device=None,
        dtype=None,
    ):
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        if config.rcps:
            self.word_embeddings = RCPSEmbedding(
                config.vocab_size,
                config.d_model,
                config.complement_map,
                **factory_kwargs,
            )
        else:
            self.word_embeddings = nn.Embedding(
                config.vocab_size, config.d_model, **factory_kwargs
            )

    def forward(self, input_ids):
        """
        input_ids: (batch, seqlen)
        """
        return self.word_embeddings(input_ids)


class WisteriaModule(nn.Module):
    """
    一个Wisteria模块，包含指定数量的层（默认8层）。
    前conv_layers_per_module层使用膨胀卷积，attn_layer_in_module层使用注意力机制。
    """
    def __init__(
        self,
        config: WisteriaConfig,
        module_idx: int,
        device=None,
        dtype=None,
    ):
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        
        self.module_idx = module_idx
        self.layers_per_module = config.layers_per_module
        self.conv_layers_per_module = config.conv_layers_per_module
        self.attn_layer_in_module = config.attn_layer_in_module
        
        # 计算该模块的层索引范围
        self.start_layer = module_idx * self.layers_per_module
        self.end_layer = (module_idx + 1) * self.layers_per_module
        
        # 创建模块内的层
        self.layers = nn.ModuleList()
        for i in range(self.layers_per_module):
            global_layer_idx = self.start_layer + i
            
            layer = create_block(
                config.d_model,
                d_intermediate=config.d_intermediate,
                attn_layer_idx=config.attn_layer_idx,
                MSC_layer_idx=config.MSC_layer_idx,
                dilation_base=config.dilation_base,
                dropout=config.dropout,
                layers_per_module=config.layers_per_module,
                conv_layers_per_module=config.conv_layers_per_module,
                attn_cfg=config.attn_cfg,
                ssm_cfg=config.ssm_cfg,
                norm_epsilon=config.norm_epsilon,
                rms_norm=config.rms_norm,
                residual_in_fp32=config.residual_in_fp32,
                fused_add_norm=config.fused_add_norm,
                layer_idx=global_layer_idx,
                bidirectional=config.bidirectional,
                bidirectional_strategy=config.bidirectional_strategy,
                bidirectional_weight_tie=config.bidirectional_weight_tie,
                rcps=config.rcps,
                # Fourier Position Embedding parameters
                use_fourier_pos_emb=config.use_fourier_pos_emb,
                fourier_max_seq_len=config.fourier_max_seq_len,
                fourier_dim=config.fourier_dim,
                fourier_init=config.fourier_init,
                fourier_init_norm_gain=config.fourier_init_norm_gain,
                fourier_separate_basis=config.fourier_separate_basis,
                fourier_separate_head=config.fourier_separate_head,
                fourier_learnable=config.fourier_learnable,
                fourier_norm=config.fourier_norm,
                fourier_ignore_zero=config.fourier_ignore_zero,
                **factory_kwargs,
            )
            self.layers.append(layer)

    def forward(self, hidden_states, residual=None, inference_params=None):
        """模块前向传播"""
        for layer in self.layers:
            hidden_states, residual = layer(
                hidden_states, residual, inference_params=inference_params
            )
        return hidden_states, residual


class WisteriaMixerModel(nn.Module):
    """
    模块化的Wisteria混合器模型，支持多个模块的组合
    """
    def __init__(
        self,
        config: WisteriaConfig,
        device=None,
        dtype=None,
    ):
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        
        self.config = config
        self.residual_in_fp32 = config.residual_in_fp32
        self.fused_add_norm = config.fused_add_norm
        
        # 添加 embeddings 层
        self.embeddings = WisteriaEmbeddings(config, **factory_kwargs)
        
        # 创建模块列表
        self.modules_list = nn.ModuleList()
        for i in range(config.n_modules):
            module = WisteriaModule(config, i, **factory_kwargs)
            self.modules_list.append(module)
        
        # 为了向后兼容，也创建一个 layers 属性
        self.layers = nn.ModuleList()
        for module in self.modules_list:
            self.layers.extend(module.layers)
        
        # 归一化层
        self.norm_f = RMSNorm(
            config.d_model,
            eps=config.norm_epsilon,
            **factory_kwargs,
        )

    def forward(self, input_ids, inputs_embeds=None, output_hidden_states=None):
        """
        模型前向传播
        """
        if inputs_embeds is None:
            inputs_embeds = self.embeddings(input_ids)
        
        hidden_states = inputs_embeds
        residual = None
        
        # 存储所有隐藏状态（如果需要）
        all_hidden_states = () if output_hidden_states else None
        
        # 通过所有模块
        for module in self.modules_list:
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)
            
            hidden_states, residual = module(
                hidden_states, residual, inference_params=None
            )
        
        # 应用最终的归一化
        if not self.fused_add_norm:
            residual = (
                (hidden_states + residual) if residual is not None else hidden_states
            )
            hidden_states = self.norm_f(residual.to(dtype=self.norm_f.weight.dtype))
        else:
            # Set prenorm=False here since we don't need the residual
            hidden_states = layer_norm_fn(
                hidden_states,
                self.norm_f.weight,
                self.norm_f.bias,
                eps=self.norm_f.eps,
                residual=residual,
                prenorm=False,
                residual_in_fp32=self.residual_in_fp32,
            )
        
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)
        
        return hidden_states, all_hidden_states

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        """为推理分配缓存"""
        return {
            i: module.allocate_inference_cache(batch_size, max_seqlen, dtype, **kwargs)
            for i, module in enumerate(self.modules_list)
        }

class WisteriaPreTrainedModel(PreTrainedModel):
    """PreTrainedModel wrapper for Wisteria backbone."""

    config_class = WisteriaConfig
    base_model_prefix = "wisteria"
    supports_gradient_checkpointing = False
    _no_split_modules = ["BiMambaWrapper"]

    def _init_weights(
        self,
        module,
        initializer_range=0.02,  # Now only used for embedding layer.
        **kwargs,
    ):
        """Adapted from: https://github.com/state-spaces/mamba/blob/main/mamba_ssm/models/mixer_seq_simple.py"""

        n_layer = self.config.n_layer
        initialized_cfg = (
            self.config.initializer_cfg
            if self.config.initializer_cfg is not None
            else {}
        )
        rescale_prenorm_residual = initialized_cfg.get("rescale_prenorm_residual", True)
        initializer_range = initialized_cfg.get("initializer_range", initializer_range)
        n_residuals_per_layer = initialized_cfg.get("n_residuals_per_layer", 1)

        if isinstance(module, nn.Linear):
            if module.bias is not None:
                if not getattr(module.bias, "_no_reinit", False):
                    nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, std=initializer_range)

        if rescale_prenorm_residual:
            # Reinitialize selected weights subject to the OpenAI GPT-2 Paper Scheme:
            #   > A modified initialization which accounts for the accumulation on the residual path with model depth.
            #   > Scale the weights of residual layers at initialization by a factor of 1/√N where N is the # of
            #   residual layers.
            #   >   -- GPT-2 :: https://openai.com/blog/better-language-models/
            #
            # Reference (Megatron-LM): https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/model/gpt_model.py
            for name, p in module.named_parameters():
                if name in ["out_proj.weight", "fc2.weight"]:
                    # Special Scaled Initialization --> There are 2 Layer Norms per Transformer Block
                    # Following Pytorch init, except scale by 1/sqrt(2 * n_layer)
                    # We need to reinit p since this code could be called multiple times
                    # Having just p *= scale would repeatedly scale it down
                    nn.init.kaiming_uniform_(p, a=math.sqrt(5))
                    with torch.no_grad():
                        p /= math.sqrt(n_residuals_per_layer * n_layer)


class Wisteria(WisteriaPreTrainedModel):
    """Wisteria model that can be instantiated using HF patterns."""

    def __init__(self, config: WisteriaConfig, device=None, dtype=None, **kwargs):
        super().__init__(config)

        if config.rcps:
            assert config.complement_map is not None, (
                "Complement map must be provided for RCPS."
            )

        # Adjust vocab size and complement maps if vocab padding is set.
        if config.vocab_size % config.pad_vocab_size_multiple != 0:
            config.vocab_size += config.pad_vocab_size_multiple - (
                config.vocab_size % config.pad_vocab_size_multiple
            )
        if config.complement_map is not None and config.vocab_size > len(
            config.complement_map
        ):
            for i in range(len(config.complement_map), config.vocab_size):
                config.complement_map[i] = i

        self.config = config
        factory_kwargs = {"device": device, "dtype": dtype}
        self.backbone = WisteriaMixerModel(config, **factory_kwargs, **kwargs)

        # Initialize weights and apply final processing
        self.post_init()

    def maybe_weight_tie_mamba(self):
        if getattr(self.config, "bidirectional", False) and getattr(
            self.config, "bidirectional_weight_tie", False
        ):
            if getattr(self.config, "rcps", False):
                for layer in self.backbone.layers:
                    # 检查是否有 submodule 和 mamba_fwd 属性（确保是 mamba 层）
                    if hasattr(layer.mixer, "submodule") and hasattr(
                        layer.mixer.submodule, "mamba_fwd"
                    ):
                        layer.mixer.submodule.mamba_rev.in_proj.weight = (
                            layer.mixer.submodule.mamba_fwd.in_proj.weight
                        )
                        layer.mixer.submodule.mamba_rev.in_proj.bias = (
                            layer.mixer.submodule.mamba_fwd.in_proj.bias
                        )
                        layer.mixer.submodule.mamba_rev.out_proj.weight = (
                            layer.mixer.submodule.mamba_fwd.out_proj.weight
                        )
                        layer.mixer.submodule.mamba_rev.out_proj.bias = (
                            layer.mixer.submodule.mamba_fwd.out_proj.bias
                        )
            else:
                for layer in self.backbone.layers:
                    # 检查是否有 mamba_fwd 属性（确保是 mamba 层）
                    if hasattr(layer.mixer, "mamba_fwd"):
                        layer.mixer.mamba_rev.in_proj.weight = (
                            layer.mixer.mamba_fwd.in_proj.weight
                        )
                        layer.mixer.mamba_rev.in_proj.bias = (
                            layer.mixer.mamba_fwd.in_proj.bias
                        )
                        layer.mixer.mamba_rev.out_proj.weight = (
                            layer.mixer.mamba_fwd.out_proj.weight
                        )
                        layer.mixer.mamba_rev.out_proj.bias = (
                            layer.mixer.mamba_fwd.out_proj.bias
                        )

    def tie_weights(self):
        self.maybe_weight_tie_mamba()

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[torch.Tensor, Tuple, BaseModelOutputWithNoAttention]:
        """HF-compatible forward method."""
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        hidden_states, all_hidden_states = self.backbone(
            input_ids,
            inputs_embeds=inputs_embeds,
            output_hidden_states=output_hidden_states,
        )
        
        if return_dict:
            return BaseModelOutputWithNoAttention(
                last_hidden_state=hidden_states,
                hidden_states=all_hidden_states if output_hidden_states else None,
            )
        elif output_hidden_states:
            return hidden_states, all_hidden_states
        else:
            return hidden_states


class WisteriaForMaskedLM(WisteriaPreTrainedModel):
    """HF-compatible Wisteria model for masked language modeling."""

    def __init__(self, config: WisteriaConfig, device=None, dtype=None, **kwargs):
        super().__init__(config, **kwargs)
        factory_kwargs = {"device": device, "dtype": dtype}
        self.wisteria = Wisteria(config, **factory_kwargs, **kwargs)
        if config.rcps:
            self.lm_head = RCPSLMHead(
                complement_map=self.config.complement_map,  # Use wisteria config as it might have been updated
                vocab_size=self.config.vocab_size,  # Use wisteria config as it might have been updated
                true_dim=config.d_model,
                dtype=dtype,
            )
        else:
            self.lm_head = nn.Linear(
                config.d_model,
                self.config.vocab_size,  # Use wisteria config as it might have been updated
                bias=False,
                **factory_kwargs,
            )

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.wisteria.backbone.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        if self.config.rcps:
            raise NotImplementedError(
                "Setting input embeddings for RCPS LM is not supported."
            )
        self.wisteria.backbone.embeddings.word_embeddings = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        """Overrides output embeddings."""
        if self.config.rcps:
            raise NotImplementedError(
                "Setting output embeddings for RCPS LM is not supported."
            )
        self.lm_head = new_embeddings

    def maybe_weight_tie_mamba(self):
        self.wisteria.maybe_weight_tie_mamba()

    def tie_weights(self):
        """Tie weights, accounting for RCPS."""
        self.maybe_weight_tie_mamba()
        if self.config.rcps:
            self.lm_head.set_weight(self.get_input_embeddings().weight)
        else:
            super().tie_weights()

    def get_decoder(self):
        """Get decoder (backbone) for the model."""
        return self.wisteria

    def set_decoder(self, decoder):
        """Set decoder (backbone) for the model."""
        self.wisteria = decoder

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        loss_weights: Optional[torch.FloatTensor] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, MaskedLMOutput]:
        """HF-compatible forward method."""

        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.wisteria(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)
        logits = logits.float()

        loss = None
        if labels is not None:
            if loss_weights is not None:
                loss = weighted_cross_entropy(
                    logits, labels, loss_weights, ignore_index=self.config.pad_token_id
                )
            else:
                loss = cross_entropy(
                    logits, labels, ignore_index=self.config.pad_token_id
                )

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return MaskedLMOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
        )


class WisteriaForSequenceClassification(WisteriaPreTrainedModel):
    def __init__(
        self,
        config: WisteriaConfig,
        pooling_strategy: str = "mean",
        conjoin_train: bool = False,
        conjoin_eval: bool = False,
        device=None,
        dtype=None,
        **kwargs,
    ):
        super().__init__(config, **kwargs)
        if pooling_strategy not in ["mean", "max", "first", "last"]:
            raise NotImplementedError(
                f"Pooling strategy `{pooling_strategy}` not implemented."
            )
        self.pooling_strategy = pooling_strategy
        factory_kwargs = {"device": device, "dtype": dtype}
        self.num_labels = kwargs.get("num_labels", config.num_labels)
        self.wisteria = Wisteria(config, **factory_kwargs, **kwargs)
        self.score = nn.Linear(config.d_model, self.num_labels, bias=False)

        self.conjoin_train = conjoin_train
        self.conjoin_eval = conjoin_eval

        # Initialize weights and apply final processing
        self.post_init()
        self.init_scorer()

    def init_scorer(self, initializer_range=0.02):
        initializer_range = (
            self.config.initializer_cfg.get("initializer_range", initializer_range)
            if self.config.initializer_cfg is not None
            else initializer_range
        )
        self.score.weight.data.normal_(std=initializer_range)

    def get_input_embeddings(self):
        return self.wisteria.backbone.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        if self.config.rcps:
            raise NotImplementedError(
                "Setting input embeddings for RCPS LM is not supported."
            )
        self.wisteria.backbone.embeddings.word_embeddings = value

    def pool_hidden_states(self, hidden_states, sequence_length_dim=1):
        """Pools hidden states along sequence length dimension."""
        if (
            self.pooling_strategy == "mean"
        ):  # Mean pooling along sequence length dimension
            return hidden_states.mean(dim=sequence_length_dim)
        if (
            self.pooling_strategy == "max"
        ):  # Max pooling along sequence length dimension
            return hidden_states.max(dim=sequence_length_dim).values
        if (
            self.pooling_strategy == "last"
        ):  # Use embedding of last token in the sequence
            return hidden_states.moveaxis(hidden_states, sequence_length_dim, 0)[
                -1, ...
            ]
        if (
            self.pooling_strategy == "first"
        ):  # Use embedding of first token in the sequence
            return hidden_states.moveaxis(hidden_states, sequence_length_dim, 0)[0, ...]

    def maybe_weight_tie_mamba(self):
        self.wisteria.maybe_weight_tie_mamba()

    def tie_weights(self):
        self.maybe_weight_tie_mamba()
        super().tie_weights()

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, SequenceClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        # Get hidden representations from the backbone
        if self.config.rcps:  # Hidden states have 2 * d_model channels for RCPS
            transformer_outputs = self.wisteria(
                input_ids,
                inputs_embeds=inputs_embeds,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
            hidden_states = torch.stack(
                [
                    transformer_outputs[0][..., : self.config.d_model],
                    torch.flip(
                        transformer_outputs[0][..., self.config.d_model :], dims=[1, 2]
                    ),
                ],
                dim=-1,
            )
        elif self.conjoin_train or (
            self.conjoin_eval and not self.training
        ):  # For conjoining / post-hoc conjoining
            assert input_ids is not None, "`input_ids` must be provided for conjoining."
            assert input_ids.ndim == 3, (
                "`input_ids` must be 3D tensor: channels corresponds to forward and rc strands."
            )
            transformer_outputs = self.wisteria(
                input_ids[..., 0],
                inputs_embeds=None,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
            transformer_outputs_rc = self.wisteria(
                input_ids[..., 1],
                inputs_embeds=None,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
            # Stack along channel dimension (dim=-1)
            hidden_states = torch.stack(
                [transformer_outputs[0], transformer_outputs_rc[0]], dim=-1
            )
        else:
            transformer_outputs = self.wisteria(
                input_ids,
                inputs_embeds=None,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
            hidden_states = transformer_outputs[0]

        # Pool and get logits
        pooled_hidden_states = self.pool_hidden_states(hidden_states)
        # Potentially run `score` twice (with parameters shared) for conjoining
        if (
            hidden_states.ndim == 4
        ):  # bsz, seq_len, hidden_dim, 2 where last channel has the stacked fwd and rc reps
            logits_fwd = self.score(pooled_hidden_states[..., 0])
            logits_rc = self.score(pooled_hidden_states[..., 1])
            logits = (logits_fwd + logits_rc) / 2
        else:
            logits = self.score(pooled_hidden_states)

        loss = None
        if labels is not None:
            labels = labels.to(logits.device)
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (
                    labels.dtype == torch.long or labels.dtype == torch.int
                ):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                if self.num_labels == 1:
                    loss = F.mse_loss(logits.squeeze(), labels.squeeze())
                else:
                    loss = F.mse_loss(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss = F.cross_entropy(
                    logits.view(-1, self.num_labels), labels.view(-1)
                )
            elif self.config.problem_type == "multi_label_classification":
                loss = F.binary_cross_entropy_with_logits(logits, labels)
        if not return_dict:
            output = (logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=transformer_outputs.hidden_states,
        )

# 删除或注释掉这个类，因为我们现在使用新的模块化版本
# class WisteriaMixerModelLegacy(nn.Module):
#     """Legacy Wisteria mixer model."""
#     ...
