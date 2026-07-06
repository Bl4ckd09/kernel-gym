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


@autotune(
    configs=[
        Config({'BLOCK_SIZE': 512}, num_warps=4),
        Config({'BLOCK_SIZE': 1024}, num_warps=4),
        Config({'BLOCK_SIZE': 2048}, num_warps=8),
        Config({'BLOCK_SIZE': 4096}, num_warps=8),
    ],
    key=['n_elements'],
)
@jit
def _bias_gelu_kernel(x_ptr, bias_ptr, out_ptr, n_elements, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements
    x = tl.load(x_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    col = offs % N
    b = tl.load(bias_ptr + col, mask=mask, other=0.0).to(tl.float32)
    xb = x + b
    SQRT_2_OVER_PI: tl.constexpr = 0.7978845608028654
    COEF: tl.constexpr = 0.044715
    inner = SQRT_2_OVER_PI * (xb + COEF * xb * xb * xb)
    # tanh via sigmoid identity: tanh(z) = 2*sigmoid(2z) - 1  (avoids relying on tl.math.tanh)
    sig = 1.0 / (1.0 + tl.exp(-2.0 * inner))
    tanh_inner = 2.0 * sig - 1.0
    out = 0.5 * xb * (1.0 + tanh_inner)
    tl.store(out_ptr + offs, out.to(x_ptr.dtype.element_ty), mask=mask)


def solution(x: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    x = x.contiguous()
    bias = bias.contiguous()
    M, N = x.shape
    n_elements = x.numel()
    out = torch.empty_like(x)
    grid = lambda meta: (cdiv(n_elements, meta['BLOCK_SIZE']),)
    _bias_gelu_kernel[grid](x, bias, out, n_elements, N)
    return out

def reference(x: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    return F.gelu(x + bias, approximate='tanh')

def make_inputs(preset, device, dtype):
    shapes = {'small': (33, 257), 'odd': (7, 1031), 'bench': (8192, 8192)}
    M, N = shapes[preset]
    return {'x': torch.randn(M, N, device=device, dtype=dtype), 'bias': torch.randn(N, device=device, dtype=dtype)}
