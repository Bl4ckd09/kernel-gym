"""t1.01 — fused bias + GELU (tanh).

Warmup: pointer arithmetic, masking, fp32 accumulation for fp16 tensors. Eager
PyTorch launches two kernels (add, then gelu) and round-trips HBM twice; one fused
kernel does a single load + single store. Pure bandwidth win — the grade is the
speedup over the eager two-op chain.

Notes lineage: "fuse the pointwise epilogue" — the most basic form of the kernel
fusion that FlashAttention and fused-softmax push to the limit.
"""

import torch
import torch.nn.functional as F

from gym import Challenge, register
from gym.tri import tl, jit, cdiv, require_triton


@jit
def _bias_gelu_kernel(x_ptr, b_ptr, out_ptr, n_elements, N, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_elements
    x = tl.load(x_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(b_ptr + offs % N, mask=mask, other=0.0).to(tl.float32)
    v = x + b
    inner = 0.7978845608028654 * (v + 0.044715 * v * v * v)
    t = 2.0 * tl.sigmoid(2.0 * inner) - 1.0  # tanh(z), stable both signs
    y = 0.5 * v * (1.0 + t)
    tl.store(out_ptr + offs, y.to(out_ptr.dtype.element_ty), mask=mask)


def solution(x: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    require_triton()
    x = x.contiguous()
    out = torch.empty_like(x)
    n = x.numel()
    grid = lambda meta: (cdiv(n, meta["BLOCK"]),)
    _bias_gelu_kernel[grid](x, bias, out, n, x.shape[-1], BLOCK=1024)
    return out


def reference(x: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    return F.gelu(x + bias, approximate="tanh")


def make_inputs(preset, device, dtype):
    shapes = {"small": (33, 257), "odd": (7, 1031), "bench": (8192, 8192)}
    M, N = shapes[preset]
    return {"x": torch.randn(M, N, device=device, dtype=dtype),
            "bias": torch.randn(N, device=device, dtype=dtype)}


register(Challenge(
    id="t1.01", name="fused bias + GELU", tier=1,
    description="Fuse bias-add and tanh-GELU into one pass; eager does two HBM round-trips.",
    sources=["@cHHillee — fusion removes intermediate HBM traffic; bandwidth is the budget"],
    make_inputs=make_inputs, reference=reference, solution=solution,
    flops=lambda i: 12 * i["x"].numel(),
    bytes=lambda i: 2 * i["x"].numel() * i["x"].element_size(),
    presets={"small": {}, "odd": {}, "bench": {}},
    grade_b=0.9, grade_a=1.1,
))
