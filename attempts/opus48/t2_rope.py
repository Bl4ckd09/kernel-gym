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

@triton.jit
def _rope_kernel(x_ptr, cos_ptr, sin_ptr, o_ptr, H, HALF, D, BLOCK_H: tl.constexpr):
    row = tl.program_id(0)
    off = tl.arange(0, BLOCK_H)
    mask = off < HALF
    cos_row = row // H
    c = tl.load(cos_ptr + cos_row * D + off, mask=mask, other=0.0).to(tl.float32)
    s = tl.load(sin_ptr + cos_row * D + off, mask=mask, other=0.0).to(tl.float32)
    x1 = tl.load(x_ptr + row * D + off, mask=mask, other=0.0).to(tl.float32)
    x2 = tl.load(x_ptr + row * D + HALF + off, mask=mask, other=0.0).to(tl.float32)
    o1 = x1 * c - x2 * s
    o2 = x2 * c + x1 * s
    tl.store(o_ptr + row * D + off, o1.to(o_ptr.dtype.element_ty), mask=mask)
    tl.store(o_ptr + row * D + HALF + off, o2.to(o_ptr.dtype.element_ty), mask=mask)


def solution(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    B, S, H, D = x.shape
    HALF = D // 2
    x = x.contiguous()
    cos = cos.contiguous()
    sin = sin.contiguous()
    out = torch.empty_like(x)
    xf = x.view(B * S * H, D)
    of = out.view(B * S * H, D)
    cf = cos.view(B * S, D)
    sf = sin.view(B * S, D)
    n_rows = B * S * H
    BLOCK_H = triton.next_power_of_2(HALF)
    grid = (n_rows,)
    _rope_kernel[grid](xf, cf, sf, of, H, HALF, D, BLOCK_H=BLOCK_H, num_warps=4)
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
