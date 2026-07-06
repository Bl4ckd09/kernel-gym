# ============================================================================
# KERNEL-GYM BLANK CHALLENGE — write the Triton kernel.
#
# Rules:
#   * Implement `solution()` below (and any @triton.jit kernels it needs).
#   * `reference()` is the ground truth; `make_inputs()` shows shapes/presets.
#     Do NOT modify either — only add your kernel code and fill in solution().
#   * solution() must return the same shape(s)/dtype(s) as reference().
#   * Accumulate reductions/matmuls in fp32. Handle non-power-of-2 shapes via
#     masking. Inputs may be non-contiguous.
#   * Graded on: correctness (hard gate, per-preset/per-dtype), then speedup
#     vs the PyTorch baseline.
# ============================================================================

import triton
import triton.language as tl
from triton import Config, autotune, cdiv

jit = triton.jit


def require_triton():
    pass

"""t5.02 — FlashAttention backward (causal): dq, dk, dv.

The frontier kernel — "the backward pass is the hard, rarely-done part" (@cloneofsimo),
"the last 20% where almost nobody operates" (@aryavs_). Given q,k,v, the forward output o,
its grad do, and the stashed per-row logsumexp L, produce all three input grads without
ever forming the (N,N) probability matrix in HBM.

The recompute trick: softmax probabilities are regenerated tile-by-tile from the scores
and L — P_ij = exp(sm_scale * q_i·k_j - L_i) — so no N*N state is stored. With the
per-row correction delta_i = sum(o_i * do_i):
    dp_ij = do_i · v_j
    ds_ij = P_ij * (dp_ij - delta_i) * sm_scale
    dv_j = sum_i P_ij do_i     dk_j = sum_i ds_ij q_i     dq_i = sum_j ds_ij k_j
Two kernels, each owning its output block so no atomics are needed: one fixes a K/V block
and streams the Q blocks (dk,dv); one fixes a Q block and streams the K/V blocks (dq).
Causal masking skips the future and applies a triangular mask on the diagonal tile.

Correctness is checked against autograd's grads — the test is the arbiter. Grade = speedup
of the whole backward vs torch.scaled_dot_product_attention's autograd backward.
"""
import math
import torch
import torch.nn.functional as F

def solution(q, k, v, o, do, L, delta, sm_scale, causal: bool=True):
    """YOUR KERNEL HERE — see rules at top of file."""
    raise NotImplementedError

def reference(q, k, v, o, do, L, delta, sm_scale, causal: bool=True):
    qf = q.float().detach().requires_grad_(True)
    kf = k.float().detach().requires_grad_(True)
    vf = v.float().detach().requires_grad_(True)
    out = F.scaled_dot_product_attention(qf, kf, vf, is_causal=causal, scale=sm_scale)
    out.backward(do.float())
    return (qf.grad.to(q.dtype), kf.grad.to(k.dtype), vf.grad.to(v.dtype))

def make_inputs(preset, device, dtype):
    shapes = {'small': (2, 4, 128, 64), 'seq': (2, 8, 512, 64), 'bench': (4, 16, 2048, 64)}
    Z, H, N, D = shapes[preset]
    sm_scale = 1.0 / math.sqrt(D)
    scale = 0.5
    q = torch.randn(Z, H, N, D, device=device, dtype=dtype) * scale
    k = torch.randn(Z, H, N, D, device=device, dtype=dtype) * scale
    v = torch.randn(Z, H, N, D, device=device, dtype=dtype) * scale
    do = torch.randn(Z, H, N, D, device=device, dtype=dtype) * scale
    s = torch.matmul(q.float(), k.float().transpose(-1, -2)) * sm_scale
    causal_mask = torch.tril(torch.ones(N, N, device=device, dtype=torch.bool))
    s = s.masked_fill(~causal_mask, float('-inf'))
    L = torch.logsumexp(s, dim=-1)
    p = torch.exp(s - L[..., None])
    o = torch.matmul(p, v.float()).to(dtype)
    delta = (o.float() * do.float()).sum(-1)
    return {'q': q, 'k': k, 'v': v, 'o': o, 'do': do, 'L': L.contiguous().to(torch.float32), 'delta': delta.contiguous().to(torch.float32), 'sm_scale': sm_scale, 'causal': True}

def _attn_bwd_flops(i):
    Z, H, N, D = i['q'].shape
    return 2.5 * 2 * 2 * Z * H * N * N * D * 0.5
