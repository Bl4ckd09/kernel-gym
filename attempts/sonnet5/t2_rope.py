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

"""t2.03 — fused RoPE apply (rotary position embedding).

Rotate the query/key head vectors by position-dependent angles. Using the GPT-NeoX /
LLaMA "rotate-half" convention: split each head dim in half, and for pairs (x1, x2)
  out1 = x1*cos - x2*sin
  out2 = x2*cos + x1*sin
cos/sin are precomputed per (position, dim). One program per (row = B*S*H) handles one
head vector; cos/sin are shared across heads so they stream from HBM cheaply.

Notes lineage: @UnslothAI — fused RoPE + MLP Triton kernels, 3x faster training. RoPE is
memory-bound: the win is doing the rotate in one pass instead of the four elementwise
ops (two muls, one neg-cat, one add) eager emits.
"""
import torch


def _next_pow2(n: int) -> int:
    return 1 << (n - 1).bit_length()


@jit
def _rope_kernel(
    x_ptr, cos_ptr, sin_ptr, out_ptr,
    S, H, D_HALF,
    stride_xb, stride_xs, stride_xh, stride_xd,
    stride_cb, stride_cs, stride_cd,
    BLOCK_HALF: tl.constexpr,
):
    row = tl.program_id(0)
    h = row % H
    tmp = row // H
    s = tmp % S
    b = tmp // S

    offs = tl.arange(0, BLOCK_HALF)
    mask = offs < D_HALF

    x_base = x_ptr + b * stride_xb + s * stride_xs + h * stride_xh
    x1 = tl.load(x_base + offs * stride_xd, mask=mask, other=0.0).to(tl.float32)
    x2 = tl.load(x_base + (offs + D_HALF) * stride_xd, mask=mask, other=0.0).to(tl.float32)

    c_base = cos_ptr + b * stride_cb + s * stride_cs
    s_base = sin_ptr + b * stride_cb + s * stride_cs
    c = tl.load(c_base + offs * stride_cd, mask=mask, other=0.0).to(tl.float32)
    sn = tl.load(s_base + offs * stride_cd, mask=mask, other=0.0).to(tl.float32)

    o1 = x1 * c - x2 * sn
    o2 = x2 * c + x1 * sn

    out_base = out_ptr + b * stride_xb + s * stride_xs + h * stride_xh
    tl.store(out_base + offs * stride_xd, o1.to(x_ptr.dtype.element_ty), mask=mask)
    tl.store(out_base + (offs + D_HALF) * stride_xd, o2.to(x_ptr.dtype.element_ty), mask=mask)


def solution(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    x = x.contiguous()
    cos = cos.contiguous()
    sin = sin.contiguous()
    B, S, H, D = x.shape
    HALF = D // 2
    out = torch.empty_like(x)
    BLOCK_HALF = _next_pow2(HALF)
    num_warps = 2 if BLOCK_HALF <= 64 else 4
    grid = (B * S * H,)
    _rope_kernel[grid](
        x, cos, sin, out,
        S, H, HALF,
        x.stride(0), x.stride(1), x.stride(2), x.stride(3),
        cos.stride(0), cos.stride(1), cos.stride(2),
        BLOCK_HALF=BLOCK_HALF,
        num_warps=num_warps,
    )
    return out

def reference(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    B, S, H, D = x.shape
    HALF = D // 2
    c = cos[..., :HALF].unsqueeze(2)
    s = sin[..., :HALF].unsqueeze(2)
    x1 = x[..., :HALF].float()
    x2 = x[..., HALF:].float()
    o1 = x1 * c - x2 * s
    o2 = x2 * c + x1 * s
    return torch.cat([o1, o2], dim=-1).to(x.dtype)

def make_inputs(preset, device, dtype):
    shapes = {'small': (2, 16, 4, 64), 'long': (1, 2048, 8, 128), 'bench': (4, 4096, 32, 128)}
    B, S, H, D = shapes[preset]
    x = torch.randn(B, S, H, D, device=device, dtype=dtype)
    pos = torch.arange(S, device=device, dtype=torch.float32)
    inv = 1.0 / 10000 ** (torch.arange(0, D // 2, device=device, dtype=torch.float32) / (D // 2))
    ang = torch.einsum('s,d->sd', pos, inv)
    ang = torch.cat([ang, ang], dim=-1)
    cos = ang.cos()[None].expand(B, S, D).to(dtype).contiguous()
    sin = ang.sin()[None].expand(B, S, D).to(dtype).contiguous()
    return {'x': x, 'cos': cos, 'sin': sin}
