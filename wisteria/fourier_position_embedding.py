"""
Fourier Position Embedding implementation for Wisteria models.
Based on the Fourier Position Embedding paper implementation.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


def _non_meta_init_device(config) -> torch.device:
    """Get the device for initialization, avoiding meta device."""
    if hasattr(config, 'init_device') and config.init_device is not None and config.init_device != "meta":
        return torch.device(config.init_device)
    else:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class RotaryEmbedding(nn.Module):
    """
    [Rotary positional embeddings (RoPE)](https://arxiv.org/abs/2104.09864).
    This is the base class for Fourier Position Embedding.
    """

    def __init__(
        self,
        config,
        cache=None,
        dim=None,
        prefix="attn",
        device=None,
        dtype=None,
    ):
        super().__init__()
        self.config = config
        self.prefix = prefix
        self.suffix = "rope"
        
        if dim is not None:
            self.dim = dim
        elif self.prefix == "attn":
            self.head_dim = self.config.d_model // getattr(self.config, 'n_heads', self.config.attn_cfg.get("num_heads", 1))
            self.dim = self.head_dim
        else:
            self.dim = self.config.d_model

        # This will be overridden in FourierEmbedding if fourier_ignore_zero is true
        inv_freq = self.get_inv_freq(self.dim, device)
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def get_inv_freq(self, dim: int, device):
        """Generate inverse frequencies."""
        rope_theta = getattr(self.config, 'rope_theta', 
                           self.config.attn_cfg.get('rotary_emb_base', 10000.0) if hasattr(self.config, 'attn_cfg') else 10000.0)
        
        inv_freq = 1.0 / (
            rope_theta ** (torch.arange(0, dim, 2, device=device, dtype=torch.float) / dim)
        )

        # This is where the frequency filtering logic from the reference model goes.
        # It's called here so both RoPE and FoPE can be configured with it if needed,
        # but it's primarily for FoPE.
        if getattr(self.config, 'use_fourier_pos_emb', False) and getattr(self.config, 'fourier_ignore_zero', False):
            max_len = getattr(self.config, 'fourier_max_seq_len', 32768)
            # Frequencies whose wavelength is longer than max_len are considered "zero" or under-trained.
            min_freq_threshold = (2 * torch.pi) / max_len
            inv_freq = inv_freq[inv_freq >= min_freq_threshold]

        return inv_freq

    def get_rotary_embedding(self, seq_len: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate rotary position embeddings."""
        with torch.autocast(device.type, enabled=False):
            seq = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
            # Use the registered buffer, which might have been filtered
            freqs = torch.einsum("t, d -> td", seq, self.inv_freq)
            emb = torch.cat((freqs, freqs), dim=-1)
            return emb.sin(), emb.cos()

    def rotate_half(self, x: torch.Tensor) -> torch.Tensor:
        """Rotate half the hidden dims of the input."""
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    def apply_rotary_pos_emb(self, pos_sin: torch.Tensor, pos_cos: torch.Tensor, t: torch.Tensor, inverse: bool = False) -> torch.Tensor:
        """Apply rotary position embedding to input tensor."""
        if not inverse:
            return (t * pos_cos) + (self.rotate_half(t) * pos_sin)
        else:
            return (t * pos_cos) - (self.rotate_half(t) * pos_sin)

    def forward(self, x: torch.Tensor, all_len: int, layer_idx: Optional[int] = None, inverse: bool = False) -> torch.Tensor:
        """Apply rotary position embedding."""
        x_ = x.float() if getattr(self.config, 'rope_full_precision', False) else x
        
        with torch.autocast(x.device.type, enabled=False):
            x_len = x_.shape[-2]
            pos_sin, pos_cos = self.get_rotary_embedding(all_len, x_.device)
            
            pos_sin = pos_sin[all_len - x_len : all_len, :]
            pos_cos = pos_cos[all_len - x_len : all_len, :]
            
            pos_sin = pos_sin.type_as(x_)
            pos_cos = pos_cos.type_as(x_)

            if self.prefix == "attn":
                pos_sin = pos_sin.unsqueeze(0).unsqueeze(0)
                pos_cos = pos_cos.unsqueeze(0).unsqueeze(0)

            x_ = self.apply_rotary_pos_emb(
                pos_sin, 
                pos_cos, 
                x_,
                inverse
            )
            
        return x_.type_as(x)


class FourierEmbedding(RotaryEmbedding):
    """
    Fourier Position Embedding (FoPE) that extends RotaryEmbedding.
    Implements "Treating Each Dimension as Multi-Frequency" and "Zero-out Under-trained Frequencies".
    """
    def __init__(self, config, cache=None, dim=None, prefix="attn", device=None, dtype=None):
        # Determine head_dim first, as it's needed for output_dim calculation
        self.head_dim = config.d_model // getattr(config, 'n_heads', config.attn_cfg.get("num_heads", 1))
        fourier_base_dim = getattr(config, 'fourier_dim', None) or self.head_dim

        # Call super().__init__ AFTER calculating dimensions
        super().__init__(config, cache=cache, dim=fourier_base_dim, prefix=prefix, device=device, dtype=dtype)
        
        self.config = config
        self.suffix = "fourier"
        
        self.separate_head = getattr(self.config, 'fourier_separate_head', True) if self.prefix == "attn" else False
        self.separate_basis = getattr(self.config, 'fourier_separate_basis', False)
        self.learnable = getattr(self.config, 'fourier_learnable', True)
        self.fourier_norm = getattr(self.config, 'fourier_norm', False)
        self.ignore_zero = getattr(self.config, 'fourier_ignore_zero', True)

        # Core logic from reference model: input_dim depends on filtered inv_freq
        self.input_dim = self.inv_freq.size(-1)
        # Output dim is half of the head dimension
        self.output_dim = self.head_dim // 2

        resolved_device = device if device is not None else _non_meta_init_device(self.config)
        factory_kwargs = {"device": resolved_device, "dtype": torch.float}

        if self.separate_head:
            n_heads = getattr(self.config, 'n_heads', self.config.attn_cfg.get("num_heads", 1))
            coef_size = (n_heads, self.input_dim, self.output_dim)
            self.coef_shape = "hDd"
            self.input_shape = "bhtD"
            self.output_shape = "bhtd"
        else:
            coef_size = (self.input_dim, self.output_dim)
            self.coef_shape = "Dd"
            self.input_shape = "btD" # Assuming batch, time, dim
            self.output_shape = "btd"

        if self.separate_basis:
            self.cos_coef = nn.Parameter(torch.empty(*coef_size, **factory_kwargs), requires_grad=self.learnable)
            self.sin_coef = nn.Parameter(torch.empty(*coef_size, **factory_kwargs), requires_grad=self.learnable)
            self.reset_parameters(self.cos_coef)
            self.reset_parameters(self.sin_coef)
        else:
            self.fourier_coefs = nn.Parameter(torch.empty(*coef_size, **factory_kwargs), requires_grad=self.learnable)
            self.reset_parameters(self.fourier_coefs)

    def get_step_eye(self, _param):
        """Helper for identity-like initialization when input_dim != output_dim."""
        _param = torch.zeros_like(_param)
        step = math.ceil(self.input_dim / self.output_dim)
        for i in range(self.output_dim):
            if i * step < self.input_dim:
                _param[..., i * step, i] = 1.0
        return _param

    def reset_parameters(self, param):
        """Initialize Fourier coefficients based on config."""
        init_method = getattr(self.config, 'fourier_init', 'eye_xavier_norm')
        gain = getattr(self.config, 'fourier_init_norm_gain', 0.3)
        
        with torch.no_grad():
            if "eye" in init_method:
                if "xavier_norm" in init_method:
                    torch.nn.init.xavier_normal_(param, gain=gain)
                elif "xavier_uniform" in init_method:
                    torch.nn.init.xavier_uniform_(param, gain=gain)
                elif "norm" in init_method:
                    torch.nn.init.normal_(param, std=gain)
                
                # Add identity-like matrix
                if self.input_dim == self.output_dim:
                    eye = torch.eye(self.input_dim, device=param.device, dtype=param.dtype)
                    param.add_(eye)
                else:
                    param.add_(self.get_step_eye(param))
            elif "xavier_norm" in init_method:
                torch.nn.init.xavier_normal_(param)
            elif "xavier_uniform" in init_method:
                torch.nn.init.xavier_uniform_(param)
            else:
                raise ValueError(f"Unsupported init method: {init_method}")

    def apply_rotary_pos_emb(self, pos_sin, pos_cos, t, inverse=False):
        """
        Applies the Fourier transformation and then the rotation.
        This is the method that implements the two core ideas.
        """
        # 1. Treating Each Dimension as Multi-Frequency
        # Base sin/cos signals are generated from filtered frequencies
        # Then they are linearly transformed by learnable coefficients.
        if self.separate_basis:
            # Ensure coefficients have the same dtype as the input tensor `t`
            sin_c, cos_c = self.sin_coef.to(t.dtype), self.cos_coef.to(t.dtype)
            if self.fourier_norm:
                sin_c = sin_c / sin_c.sum(dim=-2, keepdim=True)
                cos_c = cos_c / cos_c.sum(dim=-2, keepdim=True)
            fourier_sin = torch.einsum(f"{self.input_shape}, {self.coef_shape} -> {self.output_shape}", pos_sin, sin_c)
            fourier_cos = torch.einsum(f"{self.input_shape}, {self.coef_shape} -> {self.output_shape}", pos_cos, cos_c)
        else:
            # Ensure coefficient has the same dtype as the input tensor `t`
            coefs = self.fourier_coefs.to(t.dtype)
            if self.fourier_norm:
                coefs = coefs / coefs.sum(dim=-2, keepdim=True)
            fourier_sin = torch.einsum(f"{self.input_shape}, {self.coef_shape} -> {self.output_shape}", pos_sin, coefs)
            fourier_cos = torch.einsum(f"{self.input_shape}, {self.coef_shape} -> {self.output_shape}", pos_cos, coefs)

        # 2. Zero-out Under-trained Frequencies (Padding part)
        # After transformation, the dimension is `output_dim`. We pad it back to `head_dim // 2`.
        # The reference implementation pads with 1. This means for padded dimensions,
        # cos(padded) -> 1 and sin(padded) -> 0, effectively making them behave like RoPE at position 0.
        if self.ignore_zero:
            pad_amount = self.head_dim // 2 - self.output_dim
            if pad_amount > 0:
                # Pad with 0 for sin, 1 for cos. This is equivalent to padding freqs with 0.
                fourier_sin = F.pad(input=fourier_sin, pad=(0, pad_amount), mode="constant", value=0)
                fourier_cos = F.pad(input=fourier_cos, pad=(0, pad_amount), mode="constant", value=1)

        # Concatenate to full head dimension
        final_sin = torch.cat((fourier_sin, fourier_sin), dim=-1)
        final_cos = torch.cat((fourier_cos, fourier_cos), dim=-1)

        # Apply rotation
        if not inverse:
            return ((t * final_cos) + (self.rotate_half(t) * final_sin)).to(t.dtype)
        else:
            return ((t * final_cos) - (self.rotate_half(t) * final_sin)).to(t.dtype)

    def get_rotary_embedding(self, seq_len: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generates the base sin/cos signals for FoPE.
        Note: The actual transformation happens in `apply_rotary_pos_emb`.
        """
        with torch.autocast(device.type, enabled=False):
            t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
            # self.inv_freq is already filtered if ignore_zero=True
            freqs = torch.einsum("i,j->ij", t, self.inv_freq.to(device))
            
            # The shape needs to match the einsum in apply_rotary_pos_emb
            if self.separate_head:
                # The einsum expects "bhtD", so we need 4 dimensions.
                # The base signal is independent of batch and head, so we add singleton dimensions.
                # freqs shape: [t, D] -> pos_sin/cos shape: [1, 1, t, D]
                pos_sin = freqs.sin().unsqueeze(0).unsqueeze(0)
                pos_cos = freqs.cos().unsqueeze(0).unsqueeze(0)
            else:
                # The einsum expects "btD", so we need 3 dimensions.
                # freqs shape: [t, D] -> pos_sin/cos shape: [1, t, D]
                pos_sin = freqs.sin().unsqueeze(0)
                pos_cos = freqs.cos().unsqueeze(0)
            
            return pos_sin, pos_cos

    def forward(self, x: torch.Tensor, all_len: int, layer_idx: Optional[int] = None, inverse: bool = False) -> torch.Tensor:
        """Apply Fourier position embedding."""
        x_ = x.float() if getattr(self.config, 'rope_full_precision', False) else x
        
        with torch.autocast(x.device.type, enabled=False):
            x_len = x_.shape[-2]
            # Get base sin/cos signals. These are shaped for the einsum operation.
            pos_sin, pos_cos = self.get_rotary_embedding(all_len, x_.device)
            
            # Slice to the current length
            pos_sin = pos_sin[..., all_len - x_len : all_len, :]
            pos_cos = pos_cos[..., all_len - x_len : all_len, :]
            
            pos_sin = pos_sin.type_as(x_)
            pos_cos = pos_cos.type_as(x_)
            
            # The transformation and rotation happens here
            x_ = self.apply_rotary_pos_emb(
                pos_sin, 
                pos_cos, 
                x_,
                inverse
            )
            
        return x_.type_as(x)


class InverseFourierEmbedding(FourierEmbedding):
    """Inverse Fourier Position Embedding."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.suffix = "ifourier"
        
    def forward(self, x, all_len, layer_idx=None):
        return super().forward(x, all_len, layer_idx=layer_idx, inverse=True)


# Backward compatibility wrapper
class FourierPositionEmbedding(nn.Module):
    """
    Backward compatibility wrapper for the original interface.
    """
    
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        max_sequence_length: int = 32768,
        fourier_dim: Optional[int] = None,
        fourier_init: str = "eye_xavier_norm",
        fourier_init_norm_gain: float = 0.3,
        fourier_learnable: bool = False,
        fourier_separate_basis: bool = True,
        fourier_separate_head: bool = True,
        fourier_norm: bool = False,
        fourier_ignore_zero: bool = True,
        device=None,
        dtype=None,
    ):
        super().__init__()
        
        # Create a simplified config object to pass to the core implementation
        class SimpleConfig:
            def __init__(self):
                self.d_model = d_model
                self.n_heads = n_heads
                self.fourier_max_seq_len = max_sequence_length
                self.fourier_dim = fourier_dim
                self.fourier_init = fourier_init
                self.fourier_init_norm_gain = fourier_init_norm_gain
                self.fourier_learnable = fourier_learnable
                self.fourier_separate_basis = fourier_separate_basis
                self.fourier_separate_head = fourier_separate_head
                self.fourier_norm = fourier_norm
                self.fourier_ignore_zero = fourier_ignore_zero
                self.use_fourier_pos_emb = True
                self.rope_theta = 10000.0
                self.attn_cfg = {'num_heads': n_heads, 'rotary_emb_base': 10000.0}
        
        self.config = SimpleConfig()
        self.fourier_emb = FourierEmbedding(self.config, device=device, dtype=dtype)
    
    def forward(self, x: torch.Tensor, seq_len: Optional[int] = None) -> torch.Tensor:
        """
        Apply Fourier position embedding to input tensor.
        
        Args:
            x: Input tensor of shape (B, H, T, D) for attention
            seq_len: Sequence length (optional, will be inferred if None)
        
        Returns:
            Transformed tensor with Fourier position embedding applied
        """
        if seq_len is None:
            seq_len = x.size(-2)
        
        return self.fourier_emb(x, seq_len)
