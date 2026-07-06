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

"""t1.02 — fused SwiGLU gate: out = silu(a) * b.

The gated-MLP activation used by LLaMA/PaLM. Given the two projections a, b this is
elementwise silu(a)*b. Eager runs silu (1 pass) then multiply (another pass); the
fused kernel does one load-pair, one store. Grade = speedup over the two-op eager.

silu(x) = x * sigmoid(x). Compute in fp32 even for fp16 inputs.
"""
import torch
import torch.nn.functional as F

def solution(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """YOUR KERNEL HERE — see rules at top of file."""
    raise NotImplementedError

def reference(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return F.silu(a) * b

def make_inputs(preset, device, dtype):
    shapes = {'small': (48, 128), 'odd': (13, 777), 'bench': (8192, 11008)}
    M, N = shapes[preset]
    return {'a': torch.randn(M, N, device=device, dtype=dtype), 'b': torch.randn(M, N, device=device, dtype=dtype)}
