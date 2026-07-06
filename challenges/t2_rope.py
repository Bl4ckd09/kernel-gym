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

from gym import Challenge, register
from gym.tri import tl, jit, cdiv, require_triton


@jit
def _rope_kernel(x_ptr, cos_ptr, sin_ptr, out_ptr, n_rows, D, HALF,
                 x_row_stride, cs_row_stride, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    if row >= n_rows:
        return
    half = tl.arange(0, BLOCK)
    mask = half < HALF
    # this row belongs to position pid_pos = row // H handled by caller via cs_row_stride index
    cos = tl.load(cos_ptr + row * 0 + half, mask=mask, other=0.0)  # placeholder, overwritten below
    # NOTE: cos/sin indexed by position, passed in already broadcast per-row by the launcher
    x1 = tl.load(x_ptr + row * x_row_stride + half, mask=mask, other=0.0).to(tl.float32)
    x2 = tl.load(x_ptr + row * x_row_stride + HALF + half, mask=mask, other=0.0).to(tl.float32)
    c = tl.load(cos_ptr + row * cs_row_stride + half, mask=mask, other=0.0).to(tl.float32)
    s = tl.load(sin_ptr + row * cs_row_stride + half, mask=mask, other=0.0).to(tl.float32)
    o1 = x1 * c - x2 * s
    o2 = x2 * c + x1 * s
    tl.store(out_ptr + row * x_row_stride + half, o1.to(out_ptr.dtype.element_ty), mask=mask)
    tl.store(out_ptr + row * x_row_stride + HALF + half, o2.to(out_ptr.dtype.element_ty), mask=mask)


def solution(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """x: (B, S, H, D); cos/sin: (B, S, D) rotate-half tables (first HALF entries used)."""
    require_triton()
    B, S, H, D = x.shape
    HALF = D // 2
    xr = x.contiguous().view(B * S * H, D)
    # expand cos/sin to one row per head; only the first HALF columns are read
    cs = cos[..., :HALF].contiguous().view(B * S, HALF)
    sn = sin[..., :HALF].contiguous().view(B * S, HALF)
    cs_rep = cs.repeat_interleave(H, dim=0)
    sn_rep = sn.repeat_interleave(H, dim=0)
    out = torch.empty_like(xr)
    n_rows = B * S * H
    BLOCK = 1 << (HALF - 1).bit_length()
    _rope_kernel[(n_rows,)](xr, cs_rep, sn_rep, out, n_rows, D, HALF,
                            xr.stride(0), cs_rep.stride(0), BLOCK=BLOCK)
    return out.view(B, S, H, D)


def reference(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    B, S, H, D = x.shape
    HALF = D // 2
    c = cos[..., :HALF].unsqueeze(2)  # (B,S,1,HALF)
    s = sin[..., :HALF].unsqueeze(2)
    x1 = x[..., :HALF].float()
    x2 = x[..., HALF:].float()
    o1 = x1 * c - x2 * s
    o2 = x2 * c + x1 * s
    return torch.cat([o1, o2], dim=-1).to(x.dtype)


def make_inputs(preset, device, dtype):
    shapes = {"small": (2, 16, 4, 64), "long": (1, 2048, 8, 128), "bench": (4, 4096, 32, 128)}
    B, S, H, D = shapes[preset]
    x = torch.randn(B, S, H, D, device=device, dtype=dtype)
    pos = torch.arange(S, device=device, dtype=torch.float32)
    inv = 1.0 / (10000 ** (torch.arange(0, D // 2, device=device, dtype=torch.float32) / (D // 2)))
    ang = torch.einsum("s,d->sd", pos, inv)  # (S, HALF)
    ang = torch.cat([ang, ang], dim=-1)      # (S, D)
    cos = ang.cos()[None].expand(B, S, D).to(dtype).contiguous()
    sin = ang.sin()[None].expand(B, S, D).to(dtype).contiguous()
    return {"x": x, "cos": cos, "sin": sin}


register(Challenge(
    id="t2.03", name="fused RoPE apply", tier=2,
    description="Rotate-half RoPE on Q/K head vectors in one pass; cos/sin streamed per row.",
    sources=["@UnslothAI — fused RoPE Triton kernels, 3x faster training, -30% VRAM"],
    make_inputs=make_inputs, reference=reference, solution=solution,
    flops=lambda i: 6 * i["x"].numel(),
    bytes=lambda i: 2 * i["x"].numel() * i["x"].element_size(),
    presets={"small": {}, "long": {}, "bench": {}},
    grade_b=0.9, grade_a=1.1,
))
