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


def _next_pow2(n: int) -> int:
    return 1 << (n - 1).bit_length()


def _num_warps_for(block: int) -> int:
    if block >= 8192:
        return 16
    if block >= 2048:
        return 8
    if block >= 512:
        return 4
    return 2


@jit
def _rmsnorm_kernel(x_ptr, w_ptr, out_ptr, N, stride_m, eps, BLOCK_N: tl.constexpr):
    row = tl.program_id(0)
    col_offsets = tl.arange(0, BLOCK_N)
    mask = col_offsets < N
    row_ptr = x_ptr + row * stride_m
    x = tl.load(row_ptr + col_offsets, mask=mask, other=0.0).to(tl.float32)
    w = tl.load(w_ptr + col_offsets, mask=mask, other=0.0).to(tl.float32)
    sq_sum = tl.sum(x * x, axis=0)
    ms = sq_sum / N
    rrms = 1.0 / tl.sqrt(ms + eps)
    out = x * rrms * w
    out_row_ptr = out_ptr + row * stride_m
    tl.store(out_row_ptr + col_offsets, out.to(x_ptr.dtype.element_ty), mask=mask)


def solution(x: torch.Tensor, weight: torch.Tensor, eps: float=1e-06) -> torch.Tensor:
    x = x.contiguous()
    weight = weight.contiguous()
    M, N = x.shape
    out = torch.empty_like(x)
    BLOCK_N = _next_pow2(N)
    num_warps = _num_warps_for(BLOCK_N)
    grid = (M,)
    _rmsnorm_kernel[grid](x, weight, out, N, x.stride(0), eps, BLOCK_N=BLOCK_N, num_warps=num_warps)
    return out

def reference(x: torch.Tensor, weight: torch.Tensor, eps: float=1e-06) -> torch.Tensor:
    xf = x.float()
    ms = xf.pow(2).mean(dim=-1, keepdim=True)
    return (xf * torch.rsqrt(ms + eps)).to(x.dtype) * weight

def make_inputs(preset, device, dtype):
    shapes = {'small': (128, 256), 'wide': (32, 4096), 'bench': (8192, 4096)}
    M, N = shapes[preset]
    return {'x': torch.randn(M, N, device=device, dtype=dtype), 'weight': torch.randn(N, device=device, dtype=dtype), 'eps': 1e-06}
