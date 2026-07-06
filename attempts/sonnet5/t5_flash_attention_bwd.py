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

"""t5.02 — FlashAttention backward (causal): dq, dk, dv.

The frontier kernel — "the backward pass is the hard, rarely-done part" (@cloneofsimo),
"the last 20% where almost nobody operates" (@aryavs_). Given q,k,v, the forward output o,
its grad do, and the stashed per-row logsumexp L, produce all three input grads without
ever forming the (N,N) probability matrix in HBM.

The recompute trick: softmax probabilities are regenerated tile-by-tile from the scores
and L — P_ij = exp(sm_scale * q_i·k_j - L_i) — so no N*N state is stored. With the
per-row correction delta_i = sum(o_i * do_i):
    dp_ij = do_i · v_j
    ds_ij = P_ij * (dp_ij - delta_i) * sm_scale
    dv_j = sum_i P_ij do_i     dk_j = sum_i ds_ij q_i     dq_i = sum_j ds_ij k_j
Two kernels, each owning its output block so no atomics are needed: one fixes a K/V block
and streams the Q blocks (dk,dv); one fixes a Q block and streams the K/V blocks (dq).
Causal masking skips the future and applies a triangular mask on the diagonal tile.

Correctness is checked against autograd's grads — the test is the arbiter. Grade = speedup
of the whole backward vs torch.scaled_dot_product_attention's autograd backward.
"""
import math
import torch
import torch.nn.functional as F


@triton.jit
def _bwd_kv_kernel(
    Q, K, V, DO, L, Delta,
    DK, DV,
    stride_qz, stride_qh, stride_qm, stride_qd,
    stride_kz, stride_kh, stride_kn, stride_kd,
    stride_vz, stride_vh, stride_vn, stride_vd,
    stride_doz, stride_doh, stride_dom, stride_dod,
    stride_lz, stride_lh, stride_lm,
    stride_dz, stride_dh, stride_dm,
    stride_dkz, stride_dkh, stride_dkn, stride_dkd,
    stride_dvz, stride_dvh, stride_dvn, stride_dvd,
    H, N,
    sm_scale,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    CAUSAL: tl.constexpr,
):
    start_n = tl.program_id(0) * BLOCK_N
    off_zh = tl.program_id(1)
    off_z = off_zh // H
    off_h = off_zh % H

    Q += off_z * stride_qz + off_h * stride_qh
    K += off_z * stride_kz + off_h * stride_kh
    V += off_z * stride_vz + off_h * stride_vh
    DO += off_z * stride_doz + off_h * stride_doh
    L += off_z * stride_lz + off_h * stride_lh
    Delta += off_z * stride_dz + off_h * stride_dh
    DK += off_z * stride_dkz + off_h * stride_dkh
    DV += off_z * stride_dvz + off_h * stride_dvh

    offs_n = start_n + tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_D)
    n_mask = offs_n[:, None] < N

    k = tl.load(K + offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kd, mask=n_mask, other=0.0)
    v = tl.load(V + offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vd, mask=n_mask, other=0.0)

    dk_acc = tl.zeros([BLOCK_N, BLOCK_D], dtype=tl.float32)
    dv_acc = tl.zeros([BLOCK_N, BLOCK_D], dtype=tl.float32)

    if CAUSAL:
        lo = (start_n // BLOCK_M) * BLOCK_M
    else:
        lo = 0

    for start_m in range(lo, N, BLOCK_M):
        start_m = tl.multiple_of(start_m, BLOCK_M)
        offs_m = start_m + tl.arange(0, BLOCK_M)
        m_row_mask = offs_m < N

        q = tl.load(Q + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qd,
                    mask=m_row_mask[:, None], other=0.0)
        do = tl.load(DO + offs_m[:, None] * stride_dom + offs_d[None, :] * stride_dod,
                     mask=m_row_mask[:, None], other=0.0)
        l_i = tl.load(L + offs_m * stride_lm, mask=m_row_mask, other=0.0)
        delta_i = tl.load(Delta + offs_m * stride_dm, mask=m_row_mask, other=0.0)

        qk = tl.dot(q, tl.trans(k))
        qk = qk.to(tl.float32) * sm_scale

        valid = m_row_mask[:, None] & (offs_n[None, :] < N)
        if CAUSAL:
            valid = valid & (offs_m[:, None] >= offs_n[None, :])

        p = tl.where(valid, tl.exp(qk - l_i[:, None]), 0.0)

        dv_acc += tl.dot(tl.trans(p).to(do.dtype), do)

        dp = tl.dot(do, tl.trans(v))
        dp = dp.to(tl.float32)

        ds = p * (dp - delta_i[:, None]) * sm_scale
        ds = ds.to(q.dtype)

        dk_acc += tl.dot(tl.trans(ds), q)

    dk_ptrs = DK + offs_n[:, None] * stride_dkn + offs_d[None, :] * stride_dkd
    dv_ptrs = DV + offs_n[:, None] * stride_dvn + offs_d[None, :] * stride_dvd
    tl.store(dk_ptrs, dk_acc.to(DK.dtype.element_ty), mask=n_mask)
    tl.store(dv_ptrs, dv_acc.to(DV.dtype.element_ty), mask=n_mask)


@triton.jit
def _bwd_q_kernel(
    Q, K, V, DO, L, Delta,
    DQ,
    stride_qz, stride_qh, stride_qm, stride_qd,
    stride_kz, stride_kh, stride_kn, stride_kd,
    stride_vz, stride_vh, stride_vn, stride_vd,
    stride_doz, stride_doh, stride_dom, stride_dod,
    stride_lz, stride_lh, stride_lm,
    stride_dz, stride_dh, stride_dm,
    stride_dqz, stride_dqh, stride_dqm, stride_dqd,
    H, N,
    sm_scale,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    CAUSAL: tl.constexpr,
):
    start_m = tl.program_id(0) * BLOCK_M
    off_zh = tl.program_id(1)
    off_z = off_zh // H
    off_h = off_zh % H

    Q += off_z * stride_qz + off_h * stride_qh
    K += off_z * stride_kz + off_h * stride_kh
    V += off_z * stride_vz + off_h * stride_vh
    DO += off_z * stride_doz + off_h * stride_doh
    L += off_z * stride_lz + off_h * stride_lh
    Delta += off_z * stride_dz + off_h * stride_dh
    DQ += off_z * stride_dqz + off_h * stride_dqh

    offs_m = start_m + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_D)
    m_mask = offs_m[:, None] < N
    m_row_mask = offs_m < N

    q = tl.load(Q + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qd, mask=m_mask, other=0.0)
    do = tl.load(DO + offs_m[:, None] * stride_dom + offs_d[None, :] * stride_dod, mask=m_mask, other=0.0)
    l_i = tl.load(L + offs_m * stride_lm, mask=m_row_mask, other=0.0)
    delta_i = tl.load(Delta + offs_m * stride_dm, mask=m_row_mask, other=0.0)

    dq_acc = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)

    if CAUSAL:
        hi = start_m + BLOCK_M
        hi = tl.minimum(hi, N)
    else:
        hi = N

    for start_n in range(0, hi, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        offs_n = start_n + tl.arange(0, BLOCK_N)
        n_row_mask = offs_n < N

        k = tl.load(K + offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kd,
                    mask=n_row_mask[:, None], other=0.0)
        v = tl.load(V + offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vd,
                    mask=n_row_mask[:, None], other=0.0)

        qk = tl.dot(q, tl.trans(k))
        qk = qk.to(tl.float32) * sm_scale

        valid = m_row_mask[:, None] & n_row_mask[None, :]
        if CAUSAL:
            valid = valid & (offs_m[:, None] >= offs_n[None, :])

        p = tl.where(valid, tl.exp(qk - l_i[:, None]), 0.0)

        dp = tl.dot(do, tl.trans(v))
        dp = dp.to(tl.float32)

        ds = p * (dp - delta_i[:, None]) * sm_scale
        ds = ds.to(k.dtype)

        dq_acc += tl.dot(ds, k)

    dq_ptrs = DQ + offs_m[:, None] * stride_dqm + offs_d[None, :] * stride_dqd
    tl.store(dq_ptrs, dq_acc.to(DQ.dtype.element_ty), mask=m_mask)


def solution(q, k, v, o, do, L, delta, sm_scale, causal: bool=True):
    """YOUR KERNEL HERE — see rules at top of file."""
    Z, H, N, D = q.shape
    dk = torch.empty_like(k)
    dv = torch.empty_like(v)
    dq = torch.empty_like(q)

    BLOCK_M = 64
    BLOCK_N = 64
    BLOCK_D = D

    grid_kv = (cdiv(N, BLOCK_N), Z * H)
    _bwd_kv_kernel[grid_kv](
        q, k, v, do, L, delta,
        dk, dv,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        do.stride(0), do.stride(1), do.stride(2), do.stride(3),
        L.stride(0), L.stride(1), L.stride(2),
        delta.stride(0), delta.stride(1), delta.stride(2),
        dk.stride(0), dk.stride(1), dk.stride(2), dk.stride(3),
        dv.stride(0), dv.stride(1), dv.stride(2), dv.stride(3),
        H, N,
        sm_scale,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_D=BLOCK_D, CAUSAL=causal,
        num_warps=4, num_stages=2,
    )

    grid_q = (cdiv(N, BLOCK_M), Z * H)
    _bwd_q_kernel[grid_q](
        q, k, v, do, L, delta,
        dq,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        do.stride(0), do.stride(1), do.stride(2), do.stride(3),
        L.stride(0), L.stride(1), L.stride(2),
        delta.stride(0), delta.stride(1), delta.stride(2),
        dq.stride(0), dq.stride(1), dq.stride(2), dq.stride(3),
        H, N,
        sm_scale,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_D=BLOCK_D, CAUSAL=causal,
        num_warps=4, num_stages=2,
    )

    return dq, dk, dv

def reference(q, k, v, o, do, L, delta, sm_scale, causal: bool=True):
    qf = q.float().detach().requires_grad_(True)
    kf = k.float().detach().requires_grad_(True)
    vf = v.float().detach().requires_grad_(True)
    out = F.scaled_dot_product_attention(qf, kf, vf, is_causal=causal, scale=sm_scale)
    out.backward(do.float())
    return (qf.grad.to(q.dtype), kf.grad.to(k.dtype), vf.grad.to(v.dtype))

def make_inputs(preset, device, dtype):
    shapes = {'small': (2, 4, 128, 64), 'seq': (2, 8, 512, 64), 'bench': (4, 16, 2048, 64)}
    Z, H, N, D = shapes[preset]
    sm_scale = 1.0 / math.sqrt(D)
    scale = 0.5
    q = torch.randn(Z, H, N, D, device=device, dtype=dtype) * scale
    k = torch.randn(Z, H, N, D, device=device, dtype=dtype) * scale
    v = torch.randn(Z, H, N, D, device=device, dtype=dtype) * scale
    do = torch.randn(Z, H, N, D, device=device, dtype=dtype) * scale
    s = torch.matmul(q.float(), k.float().transpose(-1, -2)) * sm_scale
    causal_mask = torch.tril(torch.ones(N, N, device=device, dtype=torch.bool))
    s = s.masked_fill(~causal_mask, float('-inf'))
    L = torch.logsumexp(s, dim=-1)
    p = torch.exp(s - L[..., None])
    o = torch.matmul(p, v.float()).to(dtype)
    delta = (o.float() * do.float()).sum(-1)
    return {'q': q, 'k': k, 'v': v, 'o': o, 'do': do, 'L': L.contiguous().to(torch.float32), 'delta': delta.contiguous().to(torch.float32), 'sm_scale': sm_scale, 'causal': True}

def _attn_bwd_flops(i):
    Z, H, N, D = i['q'].shape
    return 2.5 * 2 * 2 * Z * H * N * N * D * 0.5
