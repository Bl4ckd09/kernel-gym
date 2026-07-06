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

from gym import Challenge, register
from gym.tri import tl, jit, require_triton


@jit
def _layernorm_fwd_kernel(x_ptr, w_ptr, b_ptr, out_ptr, mean_ptr, rstd_ptr,
                          x_row_stride, out_row_stride, n_cols, eps,
                          BLOCK: tl.constexpr):
    row = tl.program_id(0)
    col = tl.arange(0, BLOCK)
    mask = col < n_cols
    x = tl.load(x_ptr + row * x_row_stride + col, mask=mask, other=0.0).to(tl.float32)
    mean = tl.sum(x, axis=0) / n_cols
    xc = tl.where(mask, x - mean, 0.0)
    var = tl.sum(xc * xc, axis=0) / n_cols
    rstd = 1.0 / tl.sqrt(var + eps)
    w = tl.load(w_ptr + col, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(b_ptr + col, mask=mask, other=0.0).to(tl.float32)
    y = xc * rstd * w + b
    tl.store(mean_ptr + row, mean)
    tl.store(rstd_ptr + row, rstd)
    tl.store(out_ptr + row * out_row_stride + col, y.to(out_ptr.dtype.element_ty), mask=mask)


def solution(x, weight, bias, eps: float = 1e-5, return_stats: bool = False):
    require_triton()
    x = x.contiguous()
    M, N = x.shape
    out = torch.empty_like(x)
    mean = torch.empty(M, device=x.device, dtype=torch.float32)
    rstd = torch.empty(M, device=x.device, dtype=torch.float32)
    BLOCK = 1 << (N - 1).bit_length()
    nw = 16 if BLOCK >= 8192 else 8 if BLOCK >= 2048 else 4 if BLOCK >= 512 else 2
    _layernorm_fwd_kernel[(M,)](x, weight, bias, out, mean, rstd,
                                x.stride(0), out.stride(0), N, eps, BLOCK=BLOCK, num_warps=nw)
    if return_stats:
        return out, mean, rstd
    return out


def reference(x, weight, bias, eps: float = 1e-5, return_stats: bool = False):
    out = F.layer_norm(x, (x.shape[-1],), weight, bias, eps)
    if return_stats:
        xf = x.float()
        mean = xf.mean(-1)
        rstd = torch.rsqrt(xf.var(-1, unbiased=False) + eps)
        return out, mean, rstd
    return out


def make_inputs(preset, device, dtype):
    shapes = {"small": (128, 256), "wide": (32, 4096), "bench": (8192, 4096)}
    M, N = shapes[preset]
    return {"x": torch.randn(M, N, device=device, dtype=dtype),
            "weight": torch.randn(N, device=device, dtype=dtype),
            "bias": torch.randn(N, device=device, dtype=dtype),
            "eps": 1e-5}


register(Challenge(
    id="t3.03", name="LayerNorm forward (+stats)", tier=3,
    description="Mean+var in one SRAM pass; affine; stash mean/rstd for the backward.",
    sources=[
        "@aryagxr — coalesced loads + shared-mem reductions + float4 vectorization",
        "@MankyDankyBanky — fused LayerNorm one-pass, 3x over unfused",
    ],
    make_inputs=make_inputs, reference=reference, solution=solution,
    flops=lambda i: 9 * i["x"].numel(),
    bytes=lambda i: 2 * i["x"].numel() * i["x"].element_size(),
    presets={"small": {}, "wide": {}, "bench": {}},
    grade_b=0.85, grade_a=1.0,
))
