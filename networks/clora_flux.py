import torch
from library.flux_models import DoubleStreamBlock, SingleStreamBlock

axis = -1

class CLoRAAttnProcessor:
    def __init__(self, hidden_size: int, num_heads: int):
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.scale = self.head_dim ** -0.5
        self.block_sums: list[torch.Tensor] = []  # (B,L)

    def __call__(self, module, _input, _output):
        q, k, _ = _input  # (B,H,L,D)
        energy = (q * k).sum(axis) * self.scale  # (B,H,L)
        energy = energy.mean(1)  # (B,L)
        self.block_sums.append(energy.detach().cpu())
        return _output

def register_attention_hooks(model, proc):
    handles=[]
    for m in model.modules():
        if isinstance(m,(DoubleStreamBlock,SingleStreamBlock)):
            for c in m.modules():
                if c.__class__.__name__=="QKNorm":
                    handles.append(c.register_forward_hook(proc))
    return handles

def remove_attention_hooks(handles):
    for h in handles:
        h.remove()

def _resample_1d(vec: torch.Tensor, target:int):
    if vec.shape[-1]==target: return vec
    vec=vec.float()[None,None,:]
    return torch.nn.functional.interpolate(vec,size=target,mode="linear",align_corners=False).squeeze()

def compute_token_mask(sumsA:list,sumsB:list):
    max_len=max(max(t.shape[-1] for t in sumsA),max(t.shape[-1] for t in sumsB))
    to1=lambda t: t.mean(0) if t.dim()==2 else t
    vA=torch.stack([_resample_1d(to1(t),max_len) for t in sumsA]).mean(0)
    vB=torch.stack([_resample_1d(to1(t),max_len) for t in sumsB]).mean(0)
    diff=(vA-vB).abs(); diff=diff/diff.max() if diff.max()>0 else diff
    return diff  # (max_len,)

def make_mask_for_len(mask_token:torch.Tensor, seq_len:int):
    m=_resample_1d(mask_token,seq_len)
    m=(m>=0.5).float()
    maskA=m.unsqueeze(0).unsqueeze(-1)
    maskB=(1-m).unsqueeze(0).unsqueeze(-1)
    return maskA,maskB

def clora_combine(predA,predB,maskA,maskB):
    return predA*maskA + predB*maskB
