"""t1.02 — fused SwiGLU gate: out = silu(a) * b.

The gated-MLP activation used by LLaMA/PaLM. Given the two projections a, b this is
elementwise silu(a)*b. Eager runs silu (1 pass) then multiply (another pass); the
fused kernel does one load-pair, one store. Grade = speedup over the two-op eager.

silu(x) = x * sigmoid(x). Compute in fp32 even for fp16 inputs.
"""

import torch
import torch.nn.functional as F

from gym import Challenge, register
from gym.tri import tl, jit, cdiv, require_triton


@jit
def _swiglu_kernel(a_ptr, b_ptr, out_ptr, n_elements, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_elements
    a = tl.load(a_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(b_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    y = (a * tl.sigmoid(a)) * b
    tl.store(out_ptr + offs, y.to(out_ptr.dtype.element_ty), mask=mask)


def solution(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    require_triton()
    a, b = a.contiguous(), b.contiguous()
    out = torch.empty_like(a)
    n = a.numel()
    grid = lambda meta: (cdiv(n, meta["BLOCK"]),)
    _swiglu_kernel[grid](a, b, out, n, BLOCK=1024)
    return out


def reference(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return F.silu(a) * b


def make_inputs(preset, device, dtype):
    shapes = {"small": (48, 128), "odd": (13, 777), "bench": (8192, 11008)}
    M, N = shapes[preset]
    return {"a": torch.randn(M, N, device=device, dtype=dtype),
            "b": torch.randn(M, N, device=device, dtype=dtype)}


register(Challenge(
    id="t1.02", name="fused SwiGLU gate", tier=1,
    description="silu(a)*b in one kernel; the LLaMA gated-MLP activation.",
    sources=["@main_horse — gated MLPs dominate FFN FLOPs; fuse the activation"],
    make_inputs=make_inputs, reference=reference, solution=solution,
    flops=lambda i: 4 * i["a"].numel(),
    bytes=lambda i: 3 * i["a"].numel() * i["a"].element_size(),
    presets={"small": {}, "odd": {}, "bench": {}},
    grade_b=0.9, grade_a=1.1,
))
