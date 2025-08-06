"""Reusable engine to compose N LoRA models à-la CLoRA.

Exposed API
-----------
LoRAContext(multiplier:float, lora_network:LoRANetwork)
    .enable(model_or_encoders) / .disable()

AttentionCollector(hidden, heads)  # register once, holds per-LoRA block_sums
    block_sums: Dict[str,List[Tensor]]  # each (B,L)

build_masks(block_sums: Dict[str,List[Tensor]], mode="hard"|"soft", tau=0.07)
    returns Dict[str,Tensor]  # mask for each LoRA, shape (1,L,1)

fuse(pred_dict, mask_dict) -> Tensor  # weighted sum of latents
"""
from __future__ import annotations
import math
from typing import Dict, List

import torch
from library.flux_models import DoubleStreamBlock, SingleStreamBlock

axis = -1  # last dim

class LoRAContext:
    """Simple wrapper to toggle LoRA on/off during inference."""

    def __init__(self, name: str, network):
        self.name = name
        self.net = network

    def enable(self):
        self.net.set_enabled(True) if hasattr(self.net, "set_enabled") else None

    def disable(self):
        self.net.set_enabled(False) if hasattr(self.net, "set_enabled") else None


class AttentionCollector:
    def __init__(self, hidden_size: int, num_heads: int):
        self.hidden = hidden_size
        self.heads = num_heads
        self.scaler = (hidden_size // num_heads) ** -0.5
        self.block_sums: Dict[str, List[torch.Tensor]] = {}
        self.active_name = "base"
        self._handles = []

    # ---------------------------------------------------------------------
    # hook utils
    # ---------------------------------------------------------------------
    def set_active(self, name: str):
        self.active_name = name

    def clear(self):
        self.block_sums.clear()

    def attach(self, model):
        for m in model.modules():
            if isinstance(m, (DoubleStreamBlock, SingleStreamBlock)):
                for c in m.modules():
                    if c.__class__.__name__ == "QKNorm":
                        h = c.register_forward_hook(self)
                        self._handles.append(h)

    def detach(self):
        for h in self._handles:
            h.remove()
        self._handles.clear()

    # ------------------------------------------------------------------
    def __call__(self, _module, inputs, _outputs):
        q, k, _ = inputs  # (B,H,L,D)
        # raw scaled energy per head
        energy = (q * k).sum(axis) * self.scaler  # (B,H,L)
        energy = energy.mean(0).detach().cpu()  # (H,L) on CPU (batch-avg)
        self.block_sums.setdefault(self.active_name, []).append(energy)
        return _outputs


# -------------------------------------------------------------------------
# mask helpers
# -------------------------------------------------------------------------

def _normalize(x: torch.Tensor, eps: float = 1e-6):
    return x / (x.norm(dim=0, keepdim=True) + eps)

def _resample(vec: torch.Tensor, target: int) -> torch.Tensor:
    if vec.shape[-1] == target:
        return vec
    # interpolate expects 3D (N,C,W) for linear mode.
    if vec.dim() == 1:  # (L,)
        vec3 = vec.float().unsqueeze(0).unsqueeze(0)  # (1,1,L)
    else:  # (H,L)
        vec3 = vec.float().unsqueeze(0)  # (1,H,L) treating heads as channels
    out = torch.nn.functional.interpolate(vec3, size=target, mode="linear", align_corners=False)
    return out.squeeze(0)


def build_masks(block_sums: Dict[str, List[torch.Tensor]], mode: str = "soft", tau: float = 0.07) -> Dict[str, torch.Tensor]:
    """Full CLoRA mask computation (eq. 3).

    Each block entry is (H,L). We first average across blocks, yielding A_i∈ℝ^{H×L} per LoRA.
    For every token j we build the contrastive score vector s_{i,j} by applying a temperature-scaled
    softmax to the pair-wise cosine similarities between LoRAs.  For N>2 this reduces to the
    standard softmax over the cosine-similarity magnitude (same as paper §3.2).  *mode* can be
    'soft' (probability mask) or 'hard' (argmax selector)."""
    names = list(block_sums.keys())
    N = len(names)
    # 1. unify sequence length across blocks and average over blocks
    max_len = max(max(t.shape[-1] for t in lst) for lst in block_sums.values())
    A = []  # list of (H,L)
    for n in names:
        stacked = torch.stack([_resample(t, max_len) for t in block_sums[n]], 0)  # (B,H,L)
        A_i = stacked.mean(0)  # (H,L)
        A.append(A_i)
    A = torch.stack(A, 0)  # (N,H,L)

    # 2. cosine-similarity matrix per token: sim_{i,j} for i=1..N
    A_norm = _normalize(A)  # (N,H,L)
    # cosine with respect to all others: we simply compute dot with every other and sum
    # sim_i = sum_{k≠i} ⟨a_i, a_k⟩ ; then apply softmax over i (same output as paper w.r.t. pair-wise matrix)
    sim = torch.einsum("nhl,mhl->nml", A_norm, A_norm)  # (N,M,L)
    # zero self-similarity
    diag = torch.eye(N, device=sim.device)[:, :, None]
    sim = sim * (1 - diag)  # (N,N,L)
    scores = sim.sum(1)  # (N,L)

    if mode == "soft":
        logits = scores / tau
        mask = torch.softmax(logits, 0)  # (N,L)
    else:
        idx = scores.argmax(0)
        mask = torch.zeros_like(scores)
        mask[idx, torch.arange(max_len)] = 1.0

    return {n: mask[i].unsqueeze(0).unsqueeze(-1) for i, n in enumerate(names)}


def _adapt_mask(mask: torch.Tensor, target_len: int) -> torch.Tensor:
    if mask.shape[1] == target_len:
        return mask
    # mask shape (1,L,1) -> (1,1,L) for interpolate, then back
    m3 = mask.permute(0, 2, 1).float()  # (1,1,L)
    m3 = torch.nn.functional.interpolate(m3, size=target_len, mode="linear", align_corners=False)
    return m3.permute(0, 2, 1)

def fuse(preds: Dict[str, torch.Tensor], masks: Dict[str, torch.Tensor]):
    device = next(iter(preds.values())).device
    out = None
    for n, p in preds.items():
        m = masks[n].to(device)
        m = _adapt_mask(m, p.shape[1])
        out = p * m if out is None else out + p * m
    return out


