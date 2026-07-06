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

"""t1.01 — fused bias + GELU (tanh).

Warmup: pointer arithmetic, masking, fp32 accumulation for fp16 tensors. Eager
PyTorch launches two kernels (add, then gelu) and round-trips HBM twice; one fused
kernel does a single load + single store. Pure bandwidth win — the grade is the
speedup over the eager two-op chain.

Notes lineage: "fuse the pointwise epilogue" — the most basic form of the kernel
fusion that FlashAttention and fused-softmax push to the limit.
"""
import torch
import torch.nn.functional as F

def solution(x: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """YOUR KERNEL HERE — see rules at top of file."""
    raise NotImplementedError

def reference(x: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    return F.gelu(x + bias, approximate='tanh')

def make_inputs(preset, device, dtype):
    shapes = {'small': (33, 257), 'odd': (7, 1031), 'bench': (8192, 8192)}
    M, N = shapes[preset]
    return {'x': torch.randn(M, N, device=device, dtype=dtype), 'bias': torch.randn(N, device=device, dtype=dtype)}
