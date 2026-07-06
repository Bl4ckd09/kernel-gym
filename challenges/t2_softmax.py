"""t2.01 — row softmax, single pass, numerically stable.

One program per row. The row is loaded once into SRAM; max, exp, and sum are all
computed on the resident tile so HBM is touched exactly twice (read row, write row).
This is the "fused softmax" from the Triton tutorial and the conceptual seed of the
online-softmax trick that FlashAttention (t5.01) generalizes to tiles that don't fit.

Stability: subtract the row max before exp. The eager baseline `torch.softmax`
already fuses on modern PyTorch, so beating it is a real bar (grade B at 0.85x).
"""

import torch

from gym import Challenge, register
from gym.tri import tl, jit, require_triton


@jit
def _softmax_kernel(x_ptr, out_ptr, x_row_stride, out_row_stride, n_cols,
                    BLOCK: tl.constexpr):
    row = tl.program_id(0)
    col = tl.arange(0, BLOCK)
    mask = col < n_cols
    ptrs = x_ptr + row * x_row_stride + col
    x = tl.load(ptrs, mask=mask, other=-float("inf")).to(tl.float32)
    x = x - tl.max(x, axis=0)
    num = tl.exp(x)
    denom = tl.sum(num, axis=0)
    y = num / denom
    out_ptrs = out_ptr + row * out_row_stride + col
    tl.store(out_ptrs, y.to(out_ptr.dtype.element_ty), mask=mask)


def solution(x: torch.Tensor) -> torch.Tensor:
    require_triton()
    x = x.contiguous()
    M, N = x.shape
    out = torch.empty_like(x)
    BLOCK = triton_next_pow2(N)
    _softmax_kernel[(M,)](x, out, x.stride(0), out.stride(0), N,
                          BLOCK=BLOCK, num_warps=_warps_for(BLOCK))
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
    shapes = {"small": (128, 129), "wide": (64, 4096), "bench": (4096, 8192)}
    M, N = shapes[preset]
    return {"x": torch.randn(M, N, device=device, dtype=dtype) * 3.0}


register(Challenge(
    id="t2.01", name="row softmax (stable)", tier=2,
    description="One program per row, single HBM read/write, max-subtracted for stability.",
    sources=[
        "@cHHillee — fused softmax keeps the row in SRAM; memory-bound reduction",
        "@tri_dao — online/streaming softmax underlies FlashAttention",
    ],
    make_inputs=make_inputs, reference=reference, solution=solution,
    flops=lambda i: 5 * i["x"].numel(),
    bytes=lambda i: 2 * i["x"].numel() * i["x"].element_size(),
    presets={"small": {}, "wide": {}, "bench": {}},
    # PyTorch's softmax is already a fused kernel; matching it (≥0.98x) is the A bar.
    grade_b=0.85, grade_a=0.98,
))
