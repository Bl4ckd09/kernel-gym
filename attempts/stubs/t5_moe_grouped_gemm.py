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

"""t5.03 — MoE grouped GEMM: per-expert matmul over a sorted token batch.

The compute core of a Mixture-of-Experts FFN. Tokens have already been routed and sorted
by expert: `sorted_x` groups all of expert 0's tokens, then expert 1's, etc., and
`group_sizes[e]` says how many rows expert e owns. Each expert has its own weight matrix
W[e] of shape (K, N). Compute out[row] = sorted_x[row] @ W[expert_of_row] — one launch,
one variable-sized matmul per expert, no Python loop over experts and no padding every
expert to the same token count.

Why it matters: naive MoE runs E separate small matmuls (E kernel launches, terrible SM
utilization) or pads every expert to max tokens (wasted FLOPs). The grouped kernel maps a
flat grid of tiles onto the ragged group boundaries so one launch covers all experts with
no waste. The scheduling trick: precompute, per output m-tile, which expert it belongs to
and its local m-offset (`tile_expert`, `tile_m`), so every program is O(1) — no scan over
experts inside the kernel. The eager baseline does the Python-loop-of-matmuls this replaces.

Notes lineage: @tri_dao (SonicMoE) — fuse token gather with the grouped GEMM, 1.86x on
H100; @_xjdr (RDEP) — pool tokens to the owning rank so experts see D-times more tokens,
turning hundreds of tiny GEMMs into fewer fat ones; @UnslothAI — fused grouped-GEMM MoE
kernels, 12x; @PatrickToulme (CuTile) — grouped GEMM is the MoE building block.
"""
import torch
BLOCK_M = 64

def _tile_schedule(group_sizes: torch.Tensor):
    """Host-side: expand each expert into ceil(size/BLOCK_M) m-tiles -> flat schedule."""
    m_tiles = (group_sizes + BLOCK_M - 1) // BLOCK_M
    experts = torch.repeat_interleave(torch.arange(len(group_sizes), device=group_sizes.device), m_tiles)
    total = int(m_tiles.sum().item())
    ends = torch.cumsum(m_tiles, 0)
    starts_t = ends - m_tiles
    tile_m = torch.arange(total, device=group_sizes.device) - starts_t.repeat_interleave(m_tiles)
    return (experts.to(torch.int32), tile_m.to(torch.int32), total)

def solution(sorted_x, weights, group_sizes):
    """YOUR KERNEL HERE — see rules at top of file."""
    raise NotImplementedError

def reference(sorted_x, weights, group_sizes):
    out = torch.empty((sorted_x.shape[0], weights.shape[2]), device=sorted_x.device, dtype=sorted_x.dtype)
    off = 0
    for e in range(weights.shape[0]):
        n = int(group_sizes[e].item())
        if n:
            out[off:off + n] = sorted_x[off:off + n] @ weights[e]
        off += n
    return out

def make_inputs(preset, device, dtype):
    cfg = {'small': (8, 64, 128, 128), 'mid': (16, 256, 1024, 1024), 'bench': (32, 512, 4096, 4096)}
    E, avg, K, N = cfg[preset]
    torch.manual_seed(0)
    g = torch.randint(avg // 2, avg * 3 // 2 + 1, (E,), device=device)
    M = int(g.sum().item())
    return {'sorted_x': torch.randn(M, K, device=device, dtype=dtype) * 0.1, 'weights': torch.randn(E, K, N, device=device, dtype=dtype) * 0.1, 'group_sizes': g}
