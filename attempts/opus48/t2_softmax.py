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

@triton.jit
def _softmax_kernel(x_ptr, o_ptr, stride_m, N, BLOCK_N: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N
    ptrs = x_ptr + row * stride_m + cols
    x = tl.load(ptrs, mask=mask, other=-float('inf')).to(tl.float32)
    m = tl.max(x, axis=0)
    e = tl.exp(x - m)
    s = tl.sum(e, axis=0)
    y = e / s
    tl.store(o_ptr + row * stride_m + cols, y.to(o_ptr.dtype.element_ty), mask=mask)


def solution(x: torch.Tensor) -> torch.Tensor:
    x = x.contiguous()
    M, N = x.shape
    out = torch.empty_like(x)
    BLOCK_N = triton_next_pow2(N)
    num_warps = _warps_for(BLOCK_N)
    grid = (M,)
    _softmax_kernel[grid](x, out, x.stride(0), N, BLOCK_N=BLOCK_N, num_warps=num_warps)
    return out

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
