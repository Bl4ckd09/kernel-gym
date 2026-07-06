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

@triton.jit
def _bias_gelu_kernel(x_ptr, b_ptr, o_ptr, n_elements, N, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_elements
    col = offs % N
    x = tl.load(x_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(b_ptr + col, mask=mask, other=0.0).to(tl.float32)
    v = x + b
    k = 0.7978845608028654  # sqrt(2/pi)
    inner = k * (v + 0.044715 * v * v * v)
    # gelu_tanh(v) = 0.5*v*(1+tanh(inner)) = v*sigmoid(2*inner)
    g = v * tl.sigmoid(2.0 * inner)
    tl.store(o_ptr + offs, g.to(o_ptr.dtype.element_ty), mask=mask)


def solution(x: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    x = x.contiguous()
    bias = bias.contiguous()
    M, N = x.shape
    out = torch.empty_like(x)
    n_elements = x.numel()
    BLOCK = 1024
    grid = (triton.cdiv(n_elements, BLOCK),)
    _bias_gelu_kernel[grid](x, bias, out, n_elements, N, BLOCK=BLOCK, num_warps=4)
    return out

def reference(x: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    return F.gelu(x + bias, approximate='tanh')

def make_inputs(preset, device, dtype):
    shapes = {'small': (33, 257), 'odd': (7, 1031), 'bench': (8192, 8192)}
    M, N = shapes[preset]
    return {'x': torch.randn(M, N, device=device, dtype=dtype), 'bias': torch.randn(N, device=device, dtype=dtype)}
