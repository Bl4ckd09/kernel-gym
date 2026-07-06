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

"""t3.02 — INT8 GEMM with per-row / per-column dequant.

Weights are pre-quantized per output column (wq int8 + ws fp32), activations per row
(xq int8 + xs fp32). The kernel accumulates int8*int8 -> int32 in tensor cores, then
dequantizes in the epilogue: out[m,n] = acc[m,n] * xs[m] * ws[n]. This is the shape
that wins on real hardware because int8 tensor-core throughput is ~2x fp16 and the
weights are half the bytes.

Notes lineage: @maharshii — "row-scaled int8 linear, 3.1-3.5x over bf16 on RTX 4060";
@prajdabre — "the activation-outlier trap: per-tensor int8 destroys accuracy, per-row /
per-column scales recover it". Correctness here is exact vs a PyTorch int-matmul of the
SAME quantized inputs (the challenge is the kernel, not the quantizer), so tolerance is
tight — any mismatch means the accumulate or dequant is wrong.
"""
import torch

def solution(xq, wq, xs, ws):
    """YOUR KERNEL HERE — see rules at top of file."""
    raise NotImplementedError

def reference(xq, wq, xs, ws):
    acc = torch.matmul(xq.to(torch.int32).float(), wq.to(torch.int32).float())
    return (acc * xs[:, None] * ws[None, :]).to(torch.float16)

def _baseline_fp16(xq, wq, xs, ws):
    a = xq.float() * xs[:, None]
    b = wq.float() * ws[None, :]
    return a.half() @ b.half()

def make_inputs(preset, device, dtype):
    shapes = {'square': (512, 512, 512), 'odd': (129, 257, 193), 'bench': (4096, 4096, 4096)}
    M, N, K = shapes[preset]
    xq = torch.randint(-127, 128, (M, K), device=device, dtype=torch.int8)
    wq = torch.randint(-127, 128, (K, N), device=device, dtype=torch.int8)
    xs = torch.rand(M, device=device, dtype=torch.float32) * 0.02 + 0.001
    ws = torch.rand(N, device=device, dtype=torch.float32) * 0.02 + 0.001
    return {'xq': xq, 'wq': wq, 'xs': xs, 'ws': ws}
