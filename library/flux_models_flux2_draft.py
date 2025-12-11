
# region Flux2 Classes

class DoubleStreamBlockFlux2(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, mlp_ratio: float, qk_scale: float | None = None):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        head_dim = hidden_size // num_heads
        self.scale = qk_scale or head_dim**-0.5

        self.img_mod = None # Params passed from outside
        self.txt_mod = None # Params passed from outside

        self.img_norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.txt_norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)

        self.img_attn = nn.ModuleDict({
            'qkv': nn.Linear(hidden_size, hidden_size * 3, bias=True), # Checkpoint has shared qkv weight? No, separate
            'proj': nn.Linear(hidden_size, hidden_size, bias=True)
        })
        # Actually checkpoint keys are: double_blocks.0.img_attn.qkv.weight (no bias for some layers in other models, but let's check FLUX 2)
        # Based on my previous checks: double_blocks.0.img_attn.qkv.weight exists. Bias? Not checked explicitly but usually yes for qkv.
        # Wait, earlier check said "Bias keys: 0" for flux2-dev.safetensors! 
        # So NO BIAS in Linear layers!
        
        self.img_attn = AttentionSimple(hidden_size, num_heads, qk_scale)
        self.txt_attn = AttentionSimple(hidden_size, num_heads, qk_scale)

        self.img_mlp = MLPSimple(hidden_size, mlp_ratio)
        self.txt_mlp = MLPSimple(hidden_size, mlp_ratio)

        self.img_norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6) # Is there a second norm?
        # FLUX 1 architecture:
        # mod -> pre_norm -> attn -> mod_gate
        # mod -> pre_norm -> mlp -> mod_gate
        
        # FLUX 2 keys show:
        # double_blocks.0.img_attn.norm.key_norm.scale
        # double_blocks.0.img_attn.qkv.weight
        # double_blocks.0.img_attn.proj.weight
        # double_blocks.0.img_mlp.0.weight
        # double_blocks.0.img_mlp.2.weight
        
        # There are NO pre_norm weights in the checkpoint for blocks!
        # This implies standard LayerNorm (elementwise_affine=False) or it uses the modulation for norm params.
        
    def forward(self, img: Tensor, txt: Tensor, img_mod: Tensor, txt_mod: Tensor, pe: Tensor, attn_mask: Tensor = None):
        # Apply modulation (scale, shift, gate) derived globally
        # img_mod: chunk(6) -> shift_attn, scale_attn, gate_attn, shift_mlp, scale_mlp, gate_mlp
        
        shift_attn, scale_attn, gate_attn, shift_mlp, scale_mlp, gate_mlp = img_mod
        
        # Norm + Mod
        img_modulated = (1 + scale_attn) * self.img_norm1(img) + shift_attn
        
        # Attn ...
        # Wait, implementation details need to match exactly.
        pass

# Helper classes for Flux2
class AttentionSimple(nn.Module):
    def __init__(self, dim, num_heads, qk_scale=None):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=False) # No bias in FLUX 2
        self.proj = nn.Linear(dim, dim, bias=False)    # No bias in FLUX 2
        self.norm = QKNorm(head_dim)

    def forward(self, x, pe):
        B, L, _ = x.shape
        qkv = self.qkv(x)
        q, k, v = rearrange(qkv, "B L (K H D) -> K B H L D", K=3, H=self.num_heads)
        
        q, k = self.norm(q, k, v)
        
        attn = attention(q, k, v, pe=pe)
        x = self.proj(attn)
        return x

class MLPSimple(nn.Module):
    def __init__(self, dim, mlp_ratio):
        super().__init__()
        hidden = int(dim * mlp_ratio)
        self.linear1 = nn.Linear(dim, hidden, bias=False) # No bias
        self.linear2 = nn.Linear(hidden, dim, bias=False) # No bias
        self.act = nn.GELU(approximate="tanh")

    def forward(self, x):
        return self.linear2(self.act(self.linear1(x)))

# endregion
