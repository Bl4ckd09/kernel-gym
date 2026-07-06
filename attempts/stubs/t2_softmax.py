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

"""t2.01 — row softmax, single pass, numerically stable.

One program per row. The row is loaded once into SRAM; max, exp, and sum are all
computed on the resident tile so HBM is touched exactly twice (read row, write row).
This is the "fused softmax" from the Triton tutorial and the conceptual seed of the
online-softmax trick that FlashAttention (t5.01) generalizes to tiles that don't fit.

Stability: subtract the row max before exp. The eager baseline `torch.softmax`
already fuses on modern PyTorch, so beating it is a real bar (grade B at 0.85x).
"""
import torch

def solution(x: torch.Tensor) -> torch.Tensor:
    """YOUR KERNEL HERE — see rules at top of file."""
    raise NotImplementedError

def triton_next_pow2(n: int) -> int:
    return 1 << (n - 1).bit_length()

def _warps_for(block: int) -> int:
    if block >= 8192:
        return 16
    if block >= 2048:
        return 8
    if block >= 512:
        return 4
    return 2

def reference(x: torch.Tensor) -> torch.Tensor:
    return torch.softmax(x, dim=-1)

def make_inputs(preset, device, dtype):
    shapes = {'small': (128, 129), 'wide': (64, 4096), 'bench': (4096, 8192)}
    M, N = shapes[preset]
    return {'x': torch.randn(M, N, device=device, dtype=dtype) * 3.0}
