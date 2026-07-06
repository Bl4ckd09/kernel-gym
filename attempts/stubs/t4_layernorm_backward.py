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

"""t4.02 — LayerNorm backward (dx, dweight, dbias).

The gradient nobody wants to write by hand. Given dy, x, weight, and the stashed
(mean, rstd) from the forward (t3.03), produce all three grads. Two distinct patterns:

  dx  — a per-row computation. With xhat = (x-mean)*rstd and g = dy*weight,
        dx = rstd * (g - mean(g) - xhat * mean(g*xhat)). Two reductions per row over
        the feature dim, done on the SRAM-resident row. This is the memory-bound part.

  dw, db — reductions DOWN the rows (dw[n] = sum_m dy[m,n]*xhat[m,n], db[n] = sum_m dy[m,n]).
        A second kernel tiles the M dimension so each program owns a column block and
        streams all rows, accumulating in fp32. No atomics, deterministic reduction order.

Notes lineage: @vllm_project — "wrote matching custom backward passes by hand for
train/inference parity"; @cloneofsimo — "the backward pass is the hard, rarely-done part";
@aryagxr LayerNorm optimization. Grade = speedup over autograd's backward.
"""
import torch
import torch.nn.functional as F

def solution(dy, x, weight, mean, rstd):
    """YOUR KERNEL HERE — see rules at top of file."""
    raise NotImplementedError

def reference(dy, x, weight, mean, rstd):
    xf = x.float().detach().requires_grad_(True)
    w = weight.float().detach().requires_grad_(True)
    b = torch.zeros_like(w, requires_grad=True)
    y = F.layer_norm(xf, (x.shape[-1],), w, b, eps=1e-05)
    y.backward(dy.float())
    return (xf.grad.to(x.dtype), w.grad.to(weight.dtype), b.grad.to(weight.dtype))

def make_inputs(preset, device, dtype):
    shapes = {'small': (128, 256), 'wide': (64, 4096), 'bench': (8192, 4096)}
    M, N = shapes[preset]
    x = torch.randn(M, N, device=device, dtype=dtype)
    xf = x.float()
    mean = xf.mean(-1)
    rstd = torch.rsqrt(xf.var(-1, unbiased=False) + 1e-05)
    return {'dy': torch.randn(M, N, device=device, dtype=dtype), 'x': x, 'weight': torch.randn(N, device=device, dtype=dtype), 'mean': mean, 'rstd': rstd}
