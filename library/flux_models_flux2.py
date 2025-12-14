
import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor
from einops import rearrange

# Import necessary components from flux_models
from library.flux_models import (
    FluxParams, 
    QKNorm, 
    EmbedND, 
    timestep_embedding,
    attention,
    apply_rope
)

# --- Flux2 Specific Classes ---

class Modulation(nn.Module):
    """Modulation layer with keys matching Flux2 checkpoint (uses 'lin' instead of '0', '1')."""
    def __init__(self, hidden_size: int, out_features: int, bias: bool = False):
        super().__init__()
        self.lin = nn.Linear(hidden_size, out_features, bias=bias)
    
    def forward(self, x: Tensor) -> Tensor:
        return self.lin(nn.functional.silu(x))


class MLPEmbedderFlux2(nn.Module):
    """MLP Embedder without bias for Flux2."""
    def __init__(self, in_dim: int, hidden_dim: int):
        super().__init__()
        self.in_layer = nn.Linear(in_dim, hidden_dim, bias=False)
        self.silu = nn.SiLU()
        self.out_layer = nn.Linear(hidden_dim, hidden_dim, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        return self.out_layer(self.silu(self.in_layer(x)))

class MLPSimple(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, hidden_dim_out: int = None, bias: bool = False):
        super().__init__()
        if hidden_dim_out is None:
            hidden_dim_out = hidden_dim
            
        self.dim_out = hidden_dim_out
        
        # Use add_module to maintain '0', '1', '2' keys for state_dict compatibility
        self.add_module("0", nn.Linear(dim, hidden_dim, bias=bias))
        self.add_module("1", nn.GELU(approximate="tanh"))
        self.add_module("2", nn.Linear(hidden_dim_out, dim, bias=bias))

    def forward(self, x: Tensor) -> Tensor:
        x = self._modules["0"](x)
        x = self._modules["1"](x)
        if x.shape[-1] != self.dim_out:
            x = x[..., :self.dim_out]
        x = self._modules["2"](x)
        return x

class AttentionSimple(nn.Module):
    def __init__(self, dim: int, num_heads: int, qk_scale: float | None = None, bias: bool = False):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=bias)
        self.proj = nn.Linear(dim, dim, bias=bias)
        self.norm = QKNorm(head_dim)

    def forward(self, x: Tensor, pe: Tensor, attn_mask: Tensor = None) -> Tensor:
        # Standard forward for single stream or standalone
        q, k, v = self.forward_qkv(x)
        attn = attention(q, k, v, pe=pe, attn_mask=attn_mask)
        return self.forward_proj(attn)

    def forward_qkv(self, x: Tensor):
        B, L, _ = x.shape
        qkv = self.qkv(x)
        q, k, v = rearrange(qkv, "B L (K H D) -> K B H L D", K=3, H=self.num_heads)
        q, k = self.norm(q, k, v)
        return q, k, v

    def forward_proj(self, attn: Tensor):
        # Some checkpoints produce a slightly smaller head_dim/count; pad/trim to match expected dim
        in_dim = self.proj.in_features
        if attn.shape[-1] < in_dim:
            pad = in_dim - attn.shape[-1]
            attn = torch.nn.functional.pad(attn, (0, pad))
        elif attn.shape[-1] > in_dim:
            attn = attn[..., :in_dim]
        return self.proj(attn)

class DoubleStreamBlockFlux2(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, mlp_ratio: float, mlp_ratio_out: float, qk_scale: float | None = None):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        

        
        self.img_norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.txt_norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        
        self.img_attn = AttentionSimple(hidden_size, num_heads, qk_scale, bias=False)
        self.txt_attn = AttentionSimple(hidden_size, num_heads, qk_scale, bias=False)
        
        self.img_norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.txt_norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        mlp_hidden_dim_out = int(hidden_size * mlp_ratio_out)
        self.img_mlp = MLPSimple(hidden_size, mlp_hidden_dim, mlp_hidden_dim_out, bias=False)
        self.txt_mlp = MLPSimple(hidden_size, mlp_hidden_dim, mlp_hidden_dim_out, bias=False)

    def forward(self, img: Tensor, txt: Tensor, img_mod: Tensor, txt_mod: Tensor, pe: Tensor, txt_attention_mask: Tensor = None):
        img_mod1, img_mod2 = img_mod.chunk(2, dim=-1)
        txt_mod1, txt_mod2 = txt_mod.chunk(2, dim=-1)
        
        img_shift_a, img_scale_a, img_gate_a = img_mod1.chunk(3, dim=-1)
        txt_shift_a, txt_scale_a, txt_gate_a = txt_mod1.chunk(3, dim=-1)
        
        img_n = (1 + img_scale_a) * self.img_norm1(img) + img_shift_a
        txt_n = (1 + txt_scale_a) * self.txt_norm1(txt) + txt_shift_a
        
        # JOINT ATTENTION
        # 1. Get QKV for both
        img_q, img_k, img_v = self.img_attn.forward_qkv(img_n)
        txt_q, txt_k, txt_v = self.txt_attn.forward_qkv(txt_n)
        
        # 2. Concat
        q = torch.cat((txt_q, img_q), dim=2)
        k = torch.cat((txt_k, img_k), dim=2)
        v = torch.cat((txt_v, img_v), dim=2)

        # region agent log
        try:
            import json as _json, time as _time
            with open("/deployment/PictureThis/.cursor/debug.log", "a") as _f:
                _f.write(
                    _json.dumps(
                        {
                            "sessionId": "debug-session",
                            "runId": "mask-debug",
                            "hypothesisId": "H2",
                            "location": "server/sd_scripts/library/flux_models_flux2.py:125",
                            "message": "DoubleStreamBlockFlux2 mask context",
                            "data": {
                                "txt_attention_mask_shape": list(txt_attention_mask.shape) if txt_attention_mask is not None else None,
                                "txt_len": txt.shape[1],
                                "img_len": img.shape[1],
                                "q_seq_len": q.shape[2],
                            },
                            "timestamp": int(_time.time() * 1000),
                        }
                    )
                    + "\n"
                )
        except Exception:
            pass
        # endregion agent log
        
        # 3. Attend (using full PE)
        # Note: attn_mask might need adjustment if passed? 
        # Flux1 usually doesn't use masking in DoubleStreamBlock for context+img, maybe only for packing.
        # But we assume txt comes first in PE.
        attn = attention(q, k, v, pe=pe, attn_mask=txt_attention_mask)
        
        # 4. Split
        txt_attn, img_attn = attn[:, :, :txt.shape[1]], attn[:, :, txt.shape[1]:]
        
        # 5. Project and Residual
        img = img + img_gate_a * self.img_attn.forward_proj(img_attn)
        txt = txt + txt_gate_a * self.txt_attn.forward_proj(txt_attn)
        
        img_shift_m, img_scale_m, img_gate_m = img_mod2.chunk(3, dim=-1)
        txt_shift_m, txt_scale_m, txt_gate_m = txt_mod2.chunk(3, dim=-1)
        
        img_n = (1 + img_scale_m) * self.img_norm2(img) + img_shift_m
        txt_n = (1 + txt_scale_m) * self.txt_norm2(txt) + txt_shift_m
        
        img_mlp = self.img_mlp(img_n)
        txt_mlp = self.txt_mlp(txt_n)
        
        img = img + img_gate_m * img_mlp
        txt = txt + txt_gate_m * txt_mlp
        
        return img, txt

class SingleStreamBlockFlux2(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, mlp_ratio: float, mlp_ratio_out: float):
        super().__init__()
        self.hidden_dim = hidden_size
        self.num_heads = num_heads
        self.mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.mlp_hidden_dim_out = int(hidden_size * mlp_ratio_out)
        
        self.linear1 = nn.Linear(hidden_size, hidden_size * 3 + self.mlp_hidden_dim, bias=False)
        self.linear2 = nn.Linear(hidden_size + self.mlp_hidden_dim_out, hidden_size, bias=False)
        
        self.norm = QKNorm(hidden_size // num_heads)
        self.pre_norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        
        self.mlp_act = nn.GELU(approximate="tanh")
        
    def forward(self, x: Tensor, mod: Tensor, pe: Tensor, txt_attention_mask: Tensor = None):
        shift, scale, gate = mod.chunk(3, dim=-1)
        
        x_mod = (1 + scale) * self.pre_norm(x) + shift
        
        qkv, mlp = torch.split(self.linear1(x_mod), [3 * self.hidden_dim, self.mlp_hidden_dim], dim=-1)
        
        q, k, v = rearrange(qkv, "B L (K H D) -> K B H L D", K=3, H=self.num_heads)
        q, k = self.norm(q, k, v)
        
        attn = attention(q, k, v, pe=pe, attn_mask=None)
        
        mlp_out = self.mlp_act(mlp)
        mlp_out = mlp_out[..., :self.mlp_hidden_dim_out]
        
        output = self.linear2(torch.cat((attn, mlp_out), 2))
        return x + gate * output

class LastLayer(nn.Module):
    def __init__(self, hidden_size: int, patch_size: int, out_channels: int):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=False)
        # Keep Sequential for checkpoint key compatibility ('1.weight'), but without bias
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size, bias=False))

    def forward(self, x: Tensor, vec: Tensor) -> Tensor:
        shift, scale = self.adaLN_modulation(vec).chunk(2, dim=1)
        x = (1 + scale[:, None, :]) * self.norm_final(x) + shift[:, None, :]
        x = self.linear(x)
        return x

class Flux2(nn.Module):
    def __init__(self, params: FluxParams):
        super().__init__()
        self.params = params
        self.in_channels = params.in_channels
        self.out_channels = params.in_channels
        self.hidden_size = params.hidden_size
        self.num_heads = params.num_heads
        
        self.img_in = nn.Linear(self.in_channels, self.hidden_size, bias=False)
        self.time_in = MLPEmbedderFlux2(in_dim=256, hidden_dim=self.hidden_size)
        self.guidance_in = MLPEmbedderFlux2(in_dim=256, hidden_dim=self.hidden_size)
        self.txt_in = nn.Linear(params.context_in_dim, self.hidden_size, bias=False)
        
        self.pe_embedder = EmbedND(dim=self.hidden_size, theta=params.theta, axes_dim=params.axes_dim)
        
        self.double_stream_modulation_img = Modulation(self.hidden_size, 6 * self.hidden_size, bias=False)
        self.double_stream_modulation_txt = Modulation(self.hidden_size, 6 * self.hidden_size, bias=False)
        self.single_stream_modulation = Modulation(self.hidden_size, 3 * self.hidden_size, bias=False)
        
        self.double_blocks = nn.ModuleList(
            [
                DoubleStreamBlockFlux2(self.hidden_size, self.num_heads, mlp_ratio=params.mlp_ratio, mlp_ratio_out=params.mlp_ratio_out, qk_scale=params.qkv_bias)
                for _ in range(params.depth)
            ]
        )
        
        self.single_blocks = nn.ModuleList(
            [
                SingleStreamBlockFlux2(self.hidden_size, self.num_heads, mlp_ratio=params.mlp_ratio, mlp_ratio_out=params.mlp_ratio_out)
                for _ in range(params.depth_single_blocks)
            ]
        )
        
        self.final_layer = LastLayer(self.hidden_size, 1, self.out_channels)

    def forward(self, img: Tensor, img_ids: Tensor, txt: Tensor, txt_ids: Tensor, timesteps: Tensor, y: Tensor, guidance: Tensor = None, txt_attn_mask: Tensor = None):
        if img.ndim == 3:
             pass
        
        img = self.img_in(img)
        vec = self.time_in(timestep_embedding(timesteps, 256).to(img.dtype))
        if self.params.guidance_embed and guidance is not None:
             vec = vec + self.guidance_in(timestep_embedding(guidance, 256).to(img.dtype))
             
        txt = self.txt_in(txt)
        
        ids = torch.cat((txt_ids, img_ids), dim=1)
        pe = self.pe_embedder(ids)

        # region agent log
        try:
            import json as _json, time as _time
            with open("/deployment/PictureThis/.cursor/debug.log", "a") as _f:
                _f.write(
                    _json.dumps(
                        {
                            "sessionId": "debug-session",
                            "runId": "pre-fix",
                            "hypothesisId": "H1",
                            "location": "server/sd_scripts/library/flux_models_flux2.py:245",
                            "message": "Flux2 forward shapes",
                            "data": {
                                "img_shape": list(img.shape),
                                "txt_shape": list(txt.shape),
                                "img_ids_shape": list(img_ids.shape),
                                "txt_ids_shape": list(txt_ids.shape),
                                "ids_shape": list(ids.shape),
                                "pe_shape": list(pe.shape),
                            },
                            "timestamp": int(_time.time() * 1000),
                        }
                    )
                    + "\n"
                )
        except Exception:
            pass
        # endregion agent log
        
        img_mod_shared = self.double_stream_modulation_img(vec).unsqueeze(1)
        txt_mod_shared = self.double_stream_modulation_txt(vec).unsqueeze(1)
        single_mod_shared = self.single_stream_modulation(vec).unsqueeze(1)
        
        for block in self.double_blocks:
            img, txt = block(img, txt, img_mod=img_mod_shared, txt_mod=txt_mod_shared, pe=pe, txt_attention_mask=txt_attn_mask)
            
        img = torch.cat((txt, img), dim=1)
        for block in self.single_blocks:
            img = block(img, mod=single_mod_shared, pe=pe, txt_attention_mask=txt_attn_mask)
            
        img = img[:, txt.shape[1]:, ...]
        img = self.final_layer(img, vec)
        return img
