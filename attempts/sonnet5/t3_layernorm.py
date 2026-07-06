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


@jit
def _layernorm_fwd_kernel(
    x_ptr, weight_ptr, bias_ptr, out_ptr, mean_ptr, rstd_ptr,
    stride_xm, stride_om,
    N, eps,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N

    x_row_ptr = x_ptr + row * stride_xm
    x = tl.load(x_row_ptr + cols, mask=mask, other=0.0).to(tl.float32)

    mean = tl.sum(x, axis=0) / N
    diff = tl.where(mask, x - mean, 0.0)
    var = tl.sum(diff * diff, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + eps)

    x_norm = (x - mean) * rstd

    w = tl.load(weight_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(bias_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    y = x_norm * w + b

    out_row_ptr = out_ptr + row * stride_om
    tl.store(out_row_ptr + cols, y.to(out_ptr.dtype.element_ty), mask=mask)

    tl.store(mean_ptr + row, mean)
    tl.store(rstd_ptr + row, rstd)


def solution(x, weight, bias, eps: float=1e-05, return_stats: bool=False):
    """YOUR KERNEL HERE — see rules at top of file."""
    x = x.contiguous()
    weight = weight.contiguous()
    bias = bias.contiguous()
    orig_shape = x.shape
    N = orig_shape[-1]
    x2 = x.view(-1, N)
    M = x2.shape[0]

    out = torch.empty_like(x2)
    mean = torch.empty((M,), device=x.device, dtype=torch.float32)
    rstd = torch.empty((M,), device=x.device, dtype=torch.float32)

    BLOCK_N = triton.next_power_of_2(N)
    num_warps = 4
    if BLOCK_N >= 2048:
        num_warps = 8
    if BLOCK_N >= 8192:
        num_warps = 16

    grid = (M,)
    _layernorm_fwd_kernel[grid](
        x2, weight, bias, out, mean, rstd,
        x2.stride(0), out.stride(0),
        N, eps,
        BLOCK_N=BLOCK_N, num_warps=num_warps,
    )

    out = out.view(orig_shape)
    if return_stats:
        mean = mean.view(orig_shape[:-1])
        rstd = rstd.view(orig_shape[:-1])
        return out, mean, rstd
    return out

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
