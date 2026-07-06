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

"""t5.01 — FlashAttention forward (causal), online softmax, no N*N scores in HBM.

The kernel the whole set builds toward. One program owns a block of BLOCK_M queries and
streams the K/V blocks. It never materializes the (N, N) score matrix: it keeps a running
max m_i and running denominator l_i per query row, rescales the accumulator when a new
block raises the max, and only divides at the very end. Causal masking skips K blocks that
lie entirely in the future and applies a triangular mask on the diagonal block. Softmax
runs in fp32; the P@V accumulate is fp32.

This is the exact algorithm behind FA1/FA2 — same math as t2.01 softmax, generalized to
tiles that don't fit in SRAM (@tri_dao, @cHHillee, and ~every batch of the notes). FA2's
win over FA1 is precisely this loop order: Q-outer, K/V-inner, so each Q tile is loaded
once (@pranay5255). Grade = speedup over torch.scaled_dot_product_attention (which itself
dispatches a fused flash kernel), so parity is a real bar.
"""
import math
import torch
import torch.nn.functional as F

@autotune(
    configs=[
        Config({'BLOCK_M': 64, 'BLOCK_N': 64}, num_warps=4, num_stages=2),
        Config({'BLOCK_M': 128, 'BLOCK_N': 64}, num_warps=4, num_stages=3),
        Config({'BLOCK_M': 128, 'BLOCK_N': 64}, num_warps=8, num_stages=3),
        Config({'BLOCK_M': 128, 'BLOCK_N': 128}, num_warps=8, num_stages=3),
        Config({'BLOCK_M': 64, 'BLOCK_N': 64}, num_warps=4, num_stages=4),
    ],
    key=['N'],
)
@jit
def _flash_fwd(Q, K, V, O, sm_scale, Z, H, N,
               sqz, sqh, sqm, sqd,
               skz, skh, skn, skd,
               svz, svh, svn, svd,
               soz, soh, som, sod,
               HEAD_DIM: tl.constexpr, CAUSAL: tl.constexpr,
               BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    off_z = off_hz // H
    off_h = off_hz % H

    q_base = Q + off_z * sqz + off_h * sqh
    k_base = K + off_z * skz + off_h * skh
    v_base = V + off_z * svz + off_h * svh
    o_base = O + off_z * soz + off_h * soh

    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, HEAD_DIM)

    q_ptrs = q_base + offs_m[:, None] * sqm + offs_d[None, :] * sqd
    m_valid = offs_m < N
    q = tl.load(q_ptrs, mask=m_valid[:, None], other=0.0)

    m_i = tl.full([BLOCK_M], -float('inf'), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)

    if CAUSAL:
        hi = (start_m + 1) * BLOCK_M
    else:
        hi = N

    for start_n in range(0, hi, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        offs_n_c = start_n + offs_n
        n_valid = offs_n_c < N
        # K loaded transposed -> [HEAD_DIM, BLOCK_N]
        k_ptrs = k_base + offs_d[:, None] * skd + offs_n_c[None, :] * skn
        k = tl.load(k_ptrs, mask=n_valid[None, :], other=0.0)
        qk = tl.dot(q, k) * sm_scale
        if CAUSAL:
            keep = (offs_m[:, None] >= offs_n_c[None, :]) & n_valid[None, :]
        else:
            keep = n_valid[None, :] & (offs_m[:, None] < N)
        qk = tl.where(keep, qk, -float('inf'))
        m_ij = tl.maximum(m_i, tl.max(qk, 1))
        p = tl.exp(qk - m_ij[:, None])
        alpha = tl.exp(m_i - m_ij)
        l_i = l_i * alpha + tl.sum(p, 1)
        acc = acc * alpha[:, None]
        v_ptrs = v_base + offs_n_c[:, None] * svn + offs_d[None, :] * svd
        v = tl.load(v_ptrs, mask=n_valid[:, None], other=0.0)
        acc += tl.dot(p.to(v.dtype), v)
        m_i = m_ij

    acc = acc / l_i[:, None]
    o_ptrs = o_base + offs_m[:, None] * som + offs_d[None, :] * sod
    tl.store(o_ptrs, acc.to(O.dtype.element_ty), mask=m_valid[:, None])


def solution(q, k, v, causal: bool=True):
    Z, H, N, D = q.shape
    sm_scale = 1.0 / math.sqrt(D)
    o = torch.empty((Z, H, N, D), device=q.device, dtype=q.dtype)
    grid = lambda meta: (cdiv(N, meta['BLOCK_M']), Z * H)
    _flash_fwd[grid](
        q, k, v, o, sm_scale, Z, H, N,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        o.stride(0), o.stride(1), o.stride(2), o.stride(3),
        HEAD_DIM=D, CAUSAL=causal,
    )
    return o

def reference(q, k, v, causal: bool=True):
    return F.scaled_dot_product_attention(q, k, v, is_causal=causal)

def make_inputs(preset, device, dtype):
    shapes = {'small': (2, 4, 128, 64), 'seq': (2, 8, 1024, 64), 'bench': (4, 16, 4096, 64)}
    Z, H, N, D = shapes[preset]
    scale = 0.5
    return {'q': torch.randn(Z, H, N, D, device=device, dtype=dtype) * scale, 'k': torch.randn(Z, H, N, D, device=device, dtype=dtype) * scale, 'v': torch.randn(Z, H, N, D, device=device, dtype=dtype) * scale, 'causal': True}

def _attn_flops(i):
    Z, H, N, D = i['q'].shape
    return 2 * 2 * Z * H * N * N * D * 0.5
