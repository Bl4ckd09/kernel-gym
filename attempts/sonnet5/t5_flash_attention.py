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


@triton.jit
def _fwd_kernel(
    Q, K, V, O,
    stride_qz, stride_qh, stride_qm, stride_qd,
    stride_kz, stride_kh, stride_kn, stride_kd,
    stride_vz, stride_vh, stride_vn, stride_vd,
    stride_oz, stride_oh, stride_om, stride_od,
    H, N,
    sm_scale,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    CAUSAL: tl.constexpr,
):
    start_m = tl.program_id(0)
    off_zh = tl.program_id(1)
    off_z = off_zh // H
    off_h = off_zh % H

    Q += off_z * stride_qz + off_h * stride_qh
    K += off_z * stride_kz + off_h * stride_kh
    V += off_z * stride_vz + off_h * stride_vh
    O += off_z * stride_oz + off_h * stride_oh

    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_D)

    q_ptrs = Q + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qd
    q_mask = offs_m[:, None] < N
    q = tl.load(q_ptrs, mask=q_mask, other=0.0)

    m_i = tl.full([BLOCK_M], value=float("-inf"), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)

    if CAUSAL:
        hi = (start_m + 1) * BLOCK_M
        hi = tl.minimum(hi, N)
    else:
        hi = N

    for start_n in range(0, hi, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        offs_n_cur = start_n + offs_n

        k_ptrs = K + offs_n_cur[:, None] * stride_kn + offs_d[None, :] * stride_kd
        k_mask = offs_n_cur[:, None] < N
        k = tl.load(k_ptrs, mask=k_mask, other=0.0)

        qk = tl.dot(q, tl.trans(k))
        qk = qk.to(tl.float32) * sm_scale

        valid = offs_n_cur[None, :] < N
        if CAUSAL:
            valid = valid & (offs_m[:, None] >= offs_n_cur[None, :])
        qk = tl.where(valid, qk, float("-inf"))

        m_ij = tl.max(qk, axis=1)
        m_new = tl.maximum(m_i, m_ij)
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(qk - m_new[:, None])

        l_i = l_i * alpha + tl.sum(p, axis=1)
        acc = acc * alpha[:, None]

        v_ptrs = V + offs_n_cur[:, None] * stride_vn + offs_d[None, :] * stride_vd
        v_mask = offs_n_cur[:, None] < N
        v = tl.load(v_ptrs, mask=v_mask, other=0.0)

        acc += tl.dot(p.to(v.dtype), v)
        m_i = m_new

    l_i_safe = tl.where(l_i > 0, l_i, 1.0)
    acc = acc / l_i_safe[:, None]

    o_ptrs = O + offs_m[:, None] * stride_om + offs_d[None, :] * stride_od
    tl.store(o_ptrs, acc.to(O.dtype.element_ty), mask=q_mask)


def solution(q, k, v, causal: bool=True):
    """YOUR KERNEL HERE — see rules at top of file."""
    Z, H, N, D = q.shape
    o = torch.empty_like(q)
    sm_scale = 1.0 / math.sqrt(D)

    BLOCK_M = 64
    BLOCK_N = 64
    BLOCK_D = D

    grid = (cdiv(N, BLOCK_M), Z * H)

    _fwd_kernel[grid](
        q, k, v, o,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        o.stride(0), o.stride(1), o.stride(2), o.stride(3),
        H, N,
        sm_scale,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_D=BLOCK_D,
        CAUSAL=causal,
        num_warps=4, num_stages=2,
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
