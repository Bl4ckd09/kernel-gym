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

"""t4.04 — W4A16 dequant matmul: int4 weights, fp16 activations, decode-shaped.

The Marlin / Kimi-K2 inference path. Weights live in HBM as packed int4 — two values per
uint8, plus one fp16 scale per 64-element group along K — and are unpacked to fp16
in-register, never materialized. out = x @ W^T with W[n,k] = (nibble - 8) * scale[n, k//64].

Why it wins: at decode (small M) the matmul is bandwidth-bound on the WEIGHTS. fp16
weights cost 2 bytes each; packed int4 costs 0.5 + a sliver of scale — ~4x less traffic,
so the roofline says up to ~4x faster than the fp16 matmul the baseline runs. The
challenge is the in-register unpack: nibble extraction, interleave back into K order,
dequant, straight into tl.dot, fp32 accumulate.

The baseline gets the PRE-DEQUANTIZED fp16 weights for free (they're in the inputs as
`w_fp16` — your kernel should ignore them); it models a server that kept weights in fp16.

Notes lineage: @jeremyphoward/@Yuchenj_UW/@doodlestein — Kimi K2 native INT4 (1x32 block
scale) on Marlin kernels; int4 halves decode latency, advantage grows with context;
@maharshii — FP4/INT4 bit-packing: two nibbles per uint8, unpack with & 0xF and >> 4;
@elliotarledge — w4a16 GEMM written cleanly is a KernelBench-Hard problem.
"""
import torch
GROUP = 64

def solution(x, w_packed, scales, w_fp16=None):
    """YOUR KERNEL HERE — see rules at top of file."""
    raise NotImplementedError

def reference(x, w_packed, scales, w_fp16=None):
    N, Kp = w_packed.shape
    lo = (w_packed & 15).float() - 8.0
    hi = (w_packed >> 4 & 15).float() - 8.0
    w = torch.stack([lo, hi], dim=-1).reshape(N, 2 * Kp)
    w = (w * scales.repeat_interleave(GROUP, dim=1).float()).half()
    return (x.float() @ w.float().T).half()

def _baseline_fp16(x, w_packed, scales, w_fp16):
    return x @ w_fp16.T

def make_inputs(preset, device, dtype):
    shapes = {'small': (32, 256, 256), 'mid': (128, 2048, 2048), 'bench': (16, 8192, 8192)}
    M, N, K = shapes[preset]
    x = torch.randn(M, K, device=device, dtype=torch.float16)
    nib = torch.randint(0, 16, (N, K), device=device, dtype=torch.uint8)
    w_packed = nib[:, 0::2] | nib[:, 1::2] << 4
    scales = (torch.rand(N, K // GROUP, device=device) * 0.02 + 0.005).half()
    w_fp16 = ((nib.float() - 8.0) * scales.repeat_interleave(GROUP, dim=1).float()).half()
    return {'x': x, 'w_packed': w_packed, 'scales': scales, 'w_fp16': w_fp16}
