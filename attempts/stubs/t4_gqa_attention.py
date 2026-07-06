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

"""t4.03 — GQA FlashAttention forward (causal): G query heads share one KV head.

Grouped-query attention is why modern decode is affordable: with H_q query heads but only
H_kv KV heads (group size G = H_q/H_kv), the KV cache and its bandwidth shrink by G with
almost no quality loss. The kernel is t5.01's online-softmax flash forward with one twist:
program (z, h_q) reads K/V from head h_q // G. Get the indexing wrong and heads silently
attend to the wrong memory — correctness is checked against an expanded-KV reference.

The eager baseline must materialize the expanded K/V (repeat_interleave) before calling
SDPA, paying a G-times-larger KV copy; the kernel reads the compact K/V directly. That
copy is the win being graded.

Notes lineage: @rohanpaul_ai — MQA/GQA cut KV cache and bandwidth; @GoSailGlobal (CS336)
— GQA cuts decode cost ~80%, naive MHA collapses arithmetic intensity to ~1/h during
decode; @TheAhmadOsman — implement GQA and sweep #KV groups.
"""
import math
import torch
import torch.nn.functional as F

def solution(q, k, v, causal: bool=True):
    """YOUR KERNEL HERE — see rules at top of file."""
    raise NotImplementedError

def reference(q, k, v, causal: bool=True):
    G = q.shape[1] // k.shape[1]
    k_exp = k.repeat_interleave(G, dim=1)
    v_exp = v.repeat_interleave(G, dim=1)
    return F.scaled_dot_product_attention(q, k_exp, v_exp, is_causal=causal)

def make_inputs(preset, device, dtype):
    shapes = {'small': (2, 8, 2, 128, 64), 'seq': (2, 8, 2, 1024, 64), 'bench': (4, 32, 8, 4096, 64)}
    Z, Hq, Hkv, N, D = shapes[preset]
    s = 0.5
    return {'q': torch.randn(Z, Hq, N, D, device=device, dtype=dtype) * s, 'k': torch.randn(Z, Hkv, N, D, device=device, dtype=dtype) * s, 'v': torch.randn(Z, Hkv, N, D, device=device, dtype=dtype) * s, 'causal': True}

def _gqa_flops(i):
    Z, Hq, N, D = i['q'].shape
    return 2 * 2 * Z * Hq * N * N * D * 0.5
