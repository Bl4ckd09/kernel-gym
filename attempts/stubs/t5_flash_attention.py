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

"""t5.01 — FlashAttention forward (causal), online softmax, no N*N scores in HBM.

The kernel the whole set builds toward. One program owns a block of BLOCK_M queries and
streams the K/V blocks. It never materializes the (N, N) score matrix: it keeps a running
max m_i and running denominator l_i per query row, rescales the accumulator when a new
block raises the max, and only divides at the very end. Causal masking skips K blocks that
lie entirely in the future and applies a triangular mask on the diagonal block. Softmax
runs in fp32; the P@V accumulate is fp32.

This is the exact algorithm behind FA1/FA2 — same math as t2.01 softmax, generalized to
tiles that don't fit in SRAM (@tri_dao, @cHHillee, and ~every batch of the notes). FA2's
win over FA1 is precisely this loop order: Q-outer, K/V-inner, so each Q tile is loaded
once (@pranay5255). Grade = speedup over torch.scaled_dot_product_attention (which itself
dispatches a fused flash kernel), so parity is a real bar.
"""
import math
import torch
import torch.nn.functional as F

def solution(q, k, v, causal: bool=True):
    """YOUR KERNEL HERE — see rules at top of file."""
    raise NotImplementedError

def reference(q, k, v, causal: bool=True):
    return F.scaled_dot_product_attention(q, k, v, is_causal=causal)

def make_inputs(preset, device, dtype):
    shapes = {'small': (2, 4, 128, 64), 'seq': (2, 8, 1024, 64), 'bench': (4, 16, 4096, 64)}
    Z, H, N, D = shapes[preset]
    scale = 0.5
    return {'q': torch.randn(Z, H, N, D, device=device, dtype=dtype) * scale, 'k': torch.randn(Z, H, N, D, device=device, dtype=dtype) * scale, 'v': torch.randn(Z, H, N, D, device=device, dtype=dtype) * scale, 'causal': True}

def _attn_flops(i):
    Z, H, N, D = i['q'].shape
    return 2 * 2 * Z * H * N * N * D * 0.5
