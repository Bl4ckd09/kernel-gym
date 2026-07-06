"""t2.02 — RMSNorm forward.

out = x / sqrt(mean(x^2) + eps) * weight, per row. Used by LLaMA/T5 in place of
LayerNorm (no mean subtraction, no bias). One program per row: load the row into
SRAM, compute the sum of squares reduction, normalize, scale, store. HBM traffic is
one read + one write of x plus a tiny weight read — bandwidth bound.

The subtlety graded here: accumulate the sum of squares in fp32 even for fp16/bf16
inputs, or the norm is wrong for wide rows.
"""

import torch

from gym import Challenge, register
from gym.tri import tl, jit, require_triton


@jit
def _rmsnorm_kernel(x_ptr, w_ptr, out_ptr, x_row_stride, out_row_stride,
                    n_cols, eps, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    col = tl.arange(0, BLOCK)
    mask = col < n_cols
    x = tl.load(x_ptr + row * x_row_stride + col, mask=mask, other=0.0).to(tl.float32)
    w = tl.load(w_ptr + col, mask=mask, other=0.0).to(tl.float32)
    ms = tl.sum(x * x, axis=0) / n_cols
    inv = 1.0 / tl.sqrt(ms + eps)
    y = x * inv * w
    tl.store(out_ptr + row * out_row_stride + col, y.to(out_ptr.dtype.element_ty), mask=mask)


def solution(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    require_triton()
    x = x.contiguous()
    M, N = x.shape
    out = torch.empty_like(x)
    BLOCK = 1 << (N - 1).bit_length()
    nw = 16 if BLOCK >= 8192 else 8 if BLOCK >= 2048 else 4 if BLOCK >= 512 else 2
    _rmsnorm_kernel[(M,)](x, weight, out, x.stride(0), out.stride(0),
                          N, eps, BLOCK=BLOCK, num_warps=nw)
    return out


def reference(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    xf = x.float()
    ms = xf.pow(2).mean(dim=-1, keepdim=True)
    return (xf * torch.rsqrt(ms + eps)).to(x.dtype) * weight


def make_inputs(preset, device, dtype):
    shapes = {"small": (128, 256), "wide": (32, 4096), "bench": (8192, 4096)}
    M, N = shapes[preset]
    return {"x": torch.randn(M, N, device=device, dtype=dtype),
            "weight": torch.randn(N, device=device, dtype=dtype),
            "eps": 1e-6}


register(Challenge(
    id="t2.02", name="RMSNorm forward", tier=2,
    description="Per-row RMS normalize + scale; fp32 sum-of-squares accumulation.",
    sources=["@tri_dao — RMSNorm is a memory-bound row reduction; fuse scale into it"],
    make_inputs=make_inputs, reference=reference, solution=solution,
    flops=lambda i: 4 * i["x"].numel(),
    bytes=lambda i: 2 * i["x"].numel() * i["x"].element_size(),
    presets={"small": {}, "wide": {}, "bench": {}},
    grade_b=0.85, grade_a=1.0,
))
