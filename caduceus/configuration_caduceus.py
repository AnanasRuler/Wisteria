"""Caduceus config            n_layer: int = 64,
            
            # Module-based configuration
            n_modules: int = 8,                    #模块数量
            layers_per_module: int = 8,            #每个模块的层数
            conv_layers_per_module: int = 5,       #每个模块中前几层使用卷积
            attn_layer_in_module: int = 4,         #模块内注意力层的位置（0-indexed）
            
            attn_layer_idx: Optional[list] = None,      #设置哪些层使用attention，默认为None，即不使用attention
            MSC_layer_idx: Optional[list] = None,      #设置哪些层使用多尺度卷积，默认为None，即不使用MSC
            dilation_base: int = 4,     #设置膨胀卷积的基数，默认为4
            dropout: float = 0.0,
            vocab_size: int = 50277,
            ssm_cfg: Optional[dict] = None,
            attn_cfg: Optional[dict] = None,        #设置attention的配置，默认为None，即使用默认配置
            
            # Fourier Position Embedding 参数
            use_fourier_pos_emb: bool = False,     #是否使用傅里叶位置编码
            fourier_max_seq_len: int = 32768,      #傅里叶位置编码的最大序列长度
            fourier_dim: Optional[int] = None,     #傅里叶位置编码的维度
            fourier_init: str = "eye_xavier_norm", #傅里叶系数的初始化方法
            fourier_init_norm_gain: float = 0.3,  #傅里叶初始化的增益
            fourier_separate_basis: bool = True,   #是否使用分离的sin/cos基
            fourier_separate_head: bool = True,    #是否为每个注意力头使用分离的系数
            fourier_learnable: bool = False,       #傅里叶系数是否可学习
            fourier_norm: bool = False,            #是否对傅里叶系数进行归一化
            fourier_ignore_zero: bool = True,      #是否忽略零频率
            
            rms_norm: bool = True,e.

"""

from typing import Optional, Union

from transformers import PretrainedConfig


class CaduceusConfig(PretrainedConfig):
    """Config that extends the original MambaConfig with params relevant to bi-directionality and RC equivariance."""
    model_type = "caduceus"

    def __init__(
            self,
            # From original MambaConfig
            d_model: int = 2560,
            d_intermediate: int = 0,          #设置中间层(mlp)维度，默认为0，即不使用中间层           
            n_layer: int = 64,
            
            # Module-based configuration
            n_modules: int = 1,                    #模块数量
            layers_per_module: int = 8,            #每个模块的层数
            conv_layers_per_module: int = 5,       #每个模块中前几层使用卷积
            attn_layer_in_module: int = 4,         #模块内注意力层的位置（0-indexed）
            
            attn_layer_idx: Optional[list] = None,      #设置哪些层使用attention，默认为None，即不使用attention
            MSC_layer_idx: Optional[list] = None,      #设置哪些层使用多尺度卷积，默认为None，即不使用MSC
            dilation_base: int = 4,     #设置膨胀卷积的基数，默认为4
            dropout: float = 0.0,
            vocab_size: int = 50277,
            ssm_cfg: Optional[dict] = None,
            attn_cfg: Optional[dict] = None,        #设置attention的配置，默认为None，即使用默认配置
            
            # Fourier Position Embedding parameters
            use_fourier_pos_emb: bool = False,     #是否使用傅里叶位置编码
            fourier_max_seq_len: int = 32768,      #傅里叶位置编码的最大序列长度
            fourier_dim: Optional[int] = None,     #傅里叶位置编码的维度
            fourier_init: str = "eye_xavier_norm", #傅里叶系数的初始化方法
            fourier_init_norm_gain: float = 0.3,  #傅里叶初始化的增益
            fourier_separate_basis: bool = True,   #是否使用分离的sin/cos基
            fourier_separate_head: bool = True,    #是否为每个注意力头使用分离的系数
            fourier_learnable: bool = False,       #傅里叶系数是否可学习
            fourier_norm: bool = False,            #是否对傅里叶系数进行归一化
            fourier_ignore_zero: bool = True,      #是否忽略零频率
            
            rms_norm: bool = True,
            residual_in_fp32: bool = True,
            fused_add_norm: bool = True,
            pad_vocab_size_multiple: int = 8,

            # Not in original MambaConfig, but default arg in create_block in mamba_ssm repo; used in layer norm
            norm_epsilon: float = 1e-5,

            # Used in init_weights
            initializer_cfg: Optional[dict] = None,

            # Caduceus-specific params
            bidirectional: bool = True,
            bidirectional_strategy: Union[str, None] = "add",
            bidirectional_weight_tie: bool = True,
            rcps: bool = False,
            complement_map: Optional[dict] = None,  # used for RCPSEmbedding / RCPSLMHead
            **kwargs,
    ):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.d_intermediate = d_intermediate
        
        # Module-based configuration
        self.n_modules = n_modules
        self.layers_per_module = layers_per_module
        self.conv_layers_per_module = conv_layers_per_module
        self.attn_layer_in_module = attn_layer_in_module
        
        # Auto-calculate total layers and layer indices based on modules
        self.n_layer = n_modules * layers_per_module
        
        # Auto-calculate attention layer indices
        if attn_layer_idx is None:
            self.attn_layer_idx = []
            for module_idx in range(n_modules):
                base_layer = module_idx * layers_per_module
                self.attn_layer_idx.append(base_layer + attn_layer_in_module)
        else:
            self.attn_layer_idx = attn_layer_idx
            
        # Auto-calculate MSC layer indices
        if MSC_layer_idx is None:
            self.MSC_layer_idx = []
            for module_idx in range(n_modules):
                base_layer = module_idx * layers_per_module
                for conv_layer in range(conv_layers_per_module):
                    self.MSC_layer_idx.append(base_layer + conv_layer)
        else:
            self.MSC_layer_idx = MSC_layer_idx
            
        self.dilation_base = dilation_base
        self.dropout = dropout
        self.attn_cfg = attn_cfg
        
        # Fourier Position Embedding configuration
        self.use_fourier_pos_emb = use_fourier_pos_emb
        self.fourier_max_seq_len = fourier_max_seq_len
        self.fourier_dim = fourier_dim
        self.fourier_init = fourier_init
        self.fourier_init_norm_gain = fourier_init_norm_gain
        self.fourier_separate_basis = fourier_separate_basis
        self.fourier_separate_head = fourier_separate_head
        self.fourier_learnable = fourier_learnable
        self.fourier_norm = fourier_norm
        self.fourier_ignore_zero = fourier_ignore_zero
        
        self.vocab_size = vocab_size
        self.ssm_cfg = ssm_cfg
        self.rms_norm = rms_norm
        self.residual_in_fp32 = residual_in_fp32
        self.fused_add_norm = fused_add_norm
        self.pad_vocab_size_multiple = pad_vocab_size_multiple
        self.norm_epsilon = norm_epsilon
        self.initializer_cfg = initializer_cfg
        self.bidirectional = bidirectional
        self.bidirectional_strategy = bidirectional_strategy
        self.bidirectional_weight_tie = bidirectional_weight_tie
        self.rcps = rcps
        self.complement_map = complement_map
