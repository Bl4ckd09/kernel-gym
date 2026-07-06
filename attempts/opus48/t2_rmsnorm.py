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

"""t2.02 — RMSNorm forward.

out = x / sqrt(mean(x^2) + eps) * weight, per row. Used by LLaMA/T5 in place of
LayerNorm (no mean subtraction, no bias). One program per row: load the row into
SRAM, compute the sum of squares reduction, normalize, scale, store. HBM traffic is
one read + one write of x plus a tiny weight read — bandwidth bound.

The subtlety graded here: accumulate the sum of squares in fp32 even for fp16/bf16
inputs, or the norm is wrong for wide rows.
"""
import torch

@triton.jit
def _rmsnorm_kernel(x_ptr, w_ptr, o_ptr, stride_m, N, eps, BLOCK_N: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N
    x = tl.load(x_ptr + row * stride_m + cols, mask=mask, other=0.0).to(tl.float32)
    w = tl.load(w_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    ms = tl.sum(x * x, axis=0) / N
    r = 1.0 / tl.sqrt(ms + eps)
    y = x * r * w
    tl.store(o_ptr + row * stride_m + cols, y.to(o_ptr.dtype.element_ty), mask=mask)


def solution(x: torch.Tensor, weight: torch.Tensor, eps: float=1e-06) -> torch.Tensor:
    x = x.contiguous()
    weight = weight.contiguous()
    M, N = x.shape
    out = torch.empty_like(x)
    BLOCK_N = triton.next_power_of_2(N)
    if BLOCK_N >= 4096:
        num_warps = 8
    elif BLOCK_N >= 1024:
        num_warps = 4
    else:
        num_warps = 2
    grid = (M,)
    _rmsnorm_kernel[grid](x, weight, out, x.stride(0), N, eps, BLOCK_N=BLOCK_N, num_warps=num_warps)
    return out

def reference(x: torch.Tensor, weight: torch.Tensor, eps: float=1e-06) -> torch.Tensor:
    xf = x.float()
    ms = xf.pow(2).mean(dim=-1, keepdim=True)
    return (xf * torch.rsqrt(ms + eps)).to(x.dtype) * weight

def make_inputs(preset, device, dtype):
    shapes = {'small': (128, 256), 'wide': (32, 4096), 'bench': (8192, 4096)}
    M, N = shapes[preset]
    return {'x': torch.randn(M, N, device=device, dtype=dtype), 'weight': torch.randn(N, device=device, dtype=dtype), 'eps': 1e-06}
