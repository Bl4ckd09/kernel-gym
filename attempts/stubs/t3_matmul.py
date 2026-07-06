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

"""t3.01 — tiled GEMM with autotuning: C = A @ B.

The centerpiece compute-bound kernel. Program computes one BLOCK_M x BLOCK_N output
tile by streaming K in BLOCK_K chunks through SRAM, accumulating in fp32 registers.
Grouped/swizzled program ordering (GROUP_M) improves L2 reuse over naive row-major.

Notes lineage: "anatomy of high-performance matmul" (@cHHillee/@gordic_aleksa) — square
tiles maximize arithmetic intensity, register blocking hides latency; "75-80% of cuBLAS
takes weeks, the last 20% is where nobody operates" (@aryanvs_). Grade B at 0.55x cuBLAS
(torch.matmul), A at 0.75x — matching the notes' "achievable without Hopper intrinsics".
"""
import torch

def _configs():
    return [Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 64, 'GROUP_M': 8}, num_warps=8, num_stages=3), Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8}, num_warps=8, num_stages=3), Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8}, num_warps=8, num_stages=4), Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=4, num_stages=4), Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=4, num_stages=4), Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=4, num_stages=4)]

def solution(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """YOUR KERNEL HERE — see rules at top of file."""
    raise NotImplementedError

def reference(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return a @ b

def make_inputs(preset, device, dtype):
    shapes = {'square': (512, 512, 512), 'tall': (2048, 256, 1024), 'odd': (129, 257, 193), 'bench': (4096, 4096, 4096)}
    M, N, K = shapes[preset]
    s = K ** (-0.25)
    return {'a': torch.randn(M, K, device=device, dtype=dtype) * s, 'b': torch.randn(K, N, device=device, dtype=dtype) * s}
