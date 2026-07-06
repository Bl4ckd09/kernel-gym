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

"""t4.05 — sliding-window causal attention: query i attends keys j with j<=i, i-j < W.

The local-attention pattern from gpt-oss / nanochat / Mistral: alternate cheap windowed
layers with occasional global ones and the KV cost collapses. The kernel win is COMPUTE
SKIPPING, not just masking: out-of-window K/V blocks are never loaded, so work is
O(N*W) instead of O(N^2) — at N=4096, W=256 that is 16x less math than the eager path,
which must hand SDPA an explicit (N,N) band mask (arbitrary masks kill the flash
dispatch, forcing a slower backend). The mask is precomputed in the inputs for the
baseline; your kernel should ignore it and derive the band from `window` arithmetic.

Notes lineage: @karpathy (nanochat) — FA3's window_size kwarg enables alternating
local/global; @cHHillee — sliding-window is remediation #1 for linear-in-context memory;
@hamzaelshafie (gpt-oss) — banded attention + sinks; @GoSailGlobal (CS336) — 3:1
local:global interleave is the 2026 default.
"""
import math
import torch
import torch.nn.functional as F

def solution(q, k, v, window: int, mask=None):
    """YOUR KERNEL HERE — see rules at top of file."""
    raise NotImplementedError

def reference(q, k, v, window: int, mask=None):
    if mask is None:
        N = q.shape[2]
        i = torch.arange(N, device=q.device)
        mask = (i[:, None] >= i[None, :]) & (i[:, None] - i[None, :] < window)
    return F.scaled_dot_product_attention(q, k, v, attn_mask=mask)

def make_inputs(preset, device, dtype):
    shapes = {'small': (2, 4, 128, 64, 32), 'seq': (2, 8, 1024, 64, 128), 'bench': (2, 8, 4096, 64, 256)}
    Z, H, N, D, W = shapes[preset]
    s = 0.5
    i = torch.arange(N, device=device)
    mask = (i[:, None] >= i[None, :]) & (i[:, None] - i[None, :] < W)
    return {'q': torch.randn(Z, H, N, D, device=device, dtype=dtype) * s, 'k': torch.randn(Z, H, N, D, device=device, dtype=dtype) * s, 'v': torch.randn(Z, H, N, D, device=device, dtype=dtype) * s, 'window': W, 'mask': mask}

def _swa_flops(i):
    Z, H, N, D = i['q'].shape
    W = i['window']
    pairs = (N - W) * W + W * (W + 1) // 2
    return 2 * 2 * Z * H * pairs * D
