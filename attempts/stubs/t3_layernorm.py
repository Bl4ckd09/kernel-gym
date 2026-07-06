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

"""t3.03 — LayerNorm forward (mean + variance in one pass).

out = (x - mean) / sqrt(var + eps) * weight + bias, per row. Unlike RMSNorm this needs
BOTH the mean and the variance, so the program computes two reductions on the SRAM-
resident row. We also stash mean and rstd per row: the backward (t4.02) reuses them so
it never recomputes the statistics. fp32 accumulation is mandatory.

Notes lineage: @aryagxr — "LayerNorm via memory coalescing + shared-memory reductions +
vectorized loads"; @MankyDankyBanky — "fused LayerNorm (mean/var one pass, no
intermediate writes) gave 3x over unfused". Memory-bound: grade is the speedup over
eager F.layer_norm.
"""
import torch
import torch.nn.functional as F

def solution(x, weight, bias, eps: float=1e-05, return_stats: bool=False):
    """YOUR KERNEL HERE — see rules at top of file."""
    raise NotImplementedError

def reference(x, weight, bias, eps: float=1e-05, return_stats: bool=False):
    out = F.layer_norm(x, (x.shape[-1],), weight, bias, eps)
    if return_stats:
        xf = x.float()
        mean = xf.mean(-1)
        rstd = torch.rsqrt(xf.var(-1, unbiased=False) + eps)
        return (out, mean, rstd)
    return out

def make_inputs(preset, device, dtype):
    shapes = {'small': (128, 256), 'wide': (32, 4096), 'bench': (8192, 4096)}
    M, N = shapes[preset]
    return {'x': torch.randn(M, N, device=device, dtype=dtype), 'weight': torch.randn(N, device=device, dtype=dtype), 'bias': torch.randn(N, device=device, dtype=dtype), 'eps': 1e-05}
