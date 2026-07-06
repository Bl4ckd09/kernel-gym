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

_BWD_CONFIGS = [
    Config({'BLOCK': 64}, num_warps=4, num_stages=2),
    Config({'BLOCK': 128}, num_warps=8, num_stages=2),
    Config({'BLOCK': 64}, num_warps=8, num_stages=3),
    Config({'BLOCK': 128}, num_warps=4, num_stages=3),
]


@autotune(configs=_BWD_CONFIGS, key=['N'])
@jit
def _bwd_dkdv(Q, K, V, DO, DK, DV, L, Delta, sm_scale, Z, H, N,
              sqz, sqh, sqm, sqd,
              skz, skh, skn, skd,
              svz, svh, svn, svd,
              sdoz, sdoh, sdom, sdod,
              sdkz, sdkh, sdkn, sdkd,
              sdvz, sdvh, sdvn, sdvd,
              slz, slh, sln,
              sdez, sdeh, sden,
              HEAD_DIM: tl.constexpr, CAUSAL: tl.constexpr, BLOCK: tl.constexpr):
    start_n = tl.program_id(0)
    off_hz = tl.program_id(1)
    off_z = off_hz // H
    off_h = off_hz % H

    offs_n = start_n * BLOCK + tl.arange(0, BLOCK)
    offs_d = tl.arange(0, HEAD_DIM)
    n_valid = offs_n < N

    k_ptrs = K + off_z * skz + off_h * skh + offs_n[:, None] * skn + offs_d[None, :] * skd
    v_ptrs = V + off_z * svz + off_h * svh + offs_n[:, None] * svn + offs_d[None, :] * svd
    k = tl.load(k_ptrs, mask=n_valid[:, None], other=0.0)
    v = tl.load(v_ptrs, mask=n_valid[:, None], other=0.0)

    dk = tl.zeros([BLOCK, HEAD_DIM], dtype=tl.float32)
    dv = tl.zeros([BLOCK, HEAD_DIM], dtype=tl.float32)

    if CAUSAL:
        lo = start_n * BLOCK
    else:
        lo = 0

    for start_m in range(lo, N, BLOCK):
        start_m = tl.multiple_of(start_m, BLOCK)
        offs_m = start_m + tl.arange(0, BLOCK)
        m_valid = offs_m < N

        q_ptrs = Q + off_z * sqz + off_h * sqh + offs_m[:, None] * sqm + offs_d[None, :] * sqd
        do_ptrs = DO + off_z * sdoz + off_h * sdoh + offs_m[:, None] * sdom + offs_d[None, :] * sdod
        q = tl.load(q_ptrs, mask=m_valid[:, None], other=0.0)
        do = tl.load(do_ptrs, mask=m_valid[:, None], other=0.0)
        l_i = tl.load(L + off_z * slz + off_h * slh + offs_m * sln, mask=m_valid, other=0.0)
        delta_i = tl.load(Delta + off_z * sdez + off_h * sdeh + offs_m * sden, mask=m_valid, other=0.0)

        # scores^T : [BLOCK_n, BLOCK_m]
        qkT = tl.dot(k, tl.trans(q))
        pT = tl.exp(qkT * sm_scale - l_i[None, :])
        if CAUSAL:
            keep = (offs_m[None, :] >= offs_n[:, None]) & m_valid[None, :]
        else:
            keep = m_valid[None, :] & (offs_n[:, None] < N)
        pT = tl.where(keep, pT, 0.0)

        dpT = tl.dot(v, tl.trans(do))
        dsT = pT * (dpT - delta_i[None, :]) * sm_scale

        dv += tl.dot(pT.to(do.dtype), do)
        dk += tl.dot(dsT.to(q.dtype), q)

    dk_ptrs = DK + off_z * sdkz + off_h * sdkh + offs_n[:, None] * sdkn + offs_d[None, :] * sdkd
    dv_ptrs = DV + off_z * sdvz + off_h * sdvh + offs_n[:, None] * sdvn + offs_d[None, :] * sdvd
    tl.store(dk_ptrs, dk.to(DK.dtype.element_ty), mask=n_valid[:, None])
    tl.store(dv_ptrs, dv.to(DV.dtype.element_ty), mask=n_valid[:, None])


@autotune(configs=_BWD_CONFIGS, key=['N'])
@jit
def _bwd_dq(Q, K, V, DO, DQ, L, Delta, sm_scale, Z, H, N,
            sqz, sqh, sqm, sqd,
            skz, skh, skn, skd,
            svz, svh, svn, svd,
            sdoz, sdoh, sdom, sdod,
            sdqz, sdqh, sdqm, sdqd,
            slz, slh, sln,
            sdez, sdeh, sden,
            HEAD_DIM: tl.constexpr, CAUSAL: tl.constexpr, BLOCK: tl.constexpr):
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    off_z = off_hz // H
    off_h = off_hz % H

    offs_m = start_m * BLOCK + tl.arange(0, BLOCK)
    offs_d = tl.arange(0, HEAD_DIM)
    m_valid = offs_m < N

    q_ptrs = Q + off_z * sqz + off_h * sqh + offs_m[:, None] * sqm + offs_d[None, :] * sqd
    do_ptrs = DO + off_z * sdoz + off_h * sdoh + offs_m[:, None] * sdom + offs_d[None, :] * sdod
    q = tl.load(q_ptrs, mask=m_valid[:, None], other=0.0)
    do = tl.load(do_ptrs, mask=m_valid[:, None], other=0.0)
    l_i = tl.load(L + off_z * slz + off_h * slh + offs_m * sln, mask=m_valid, other=0.0)
    delta_i = tl.load(Delta + off_z * sdez + off_h * sdeh + offs_m * sden, mask=m_valid, other=0.0)

    dq = tl.zeros([BLOCK, HEAD_DIM], dtype=tl.float32)

    if CAUSAL:
        hi = (start_m + 1) * BLOCK
    else:
        hi = N

    for start_n in range(0, hi, BLOCK):
        start_n = tl.multiple_of(start_n, BLOCK)
        offs_n = start_n + tl.arange(0, BLOCK)
        n_valid = offs_n < N

        k_ptrs = K + off_z * skz + off_h * skh + offs_n[:, None] * skn + offs_d[None, :] * skd
        v_ptrs = V + off_z * svz + off_h * svh + offs_n[:, None] * svn + offs_d[None, :] * svd
        k = tl.load(k_ptrs, mask=n_valid[:, None], other=0.0)
        v = tl.load(v_ptrs, mask=n_valid[:, None], other=0.0)

        # scores : [BLOCK_m, BLOCK_n]
        qk = tl.dot(q, tl.trans(k))
        p = tl.exp(qk * sm_scale - l_i[:, None])
        if CAUSAL:
            keep = (offs_m[:, None] >= offs_n[None, :]) & n_valid[None, :]
        else:
            keep = n_valid[None, :] & m_valid[:, None]
        p = tl.where(keep, p, 0.0)

        dp = tl.dot(do, tl.trans(v))
        ds = p * (dp - delta_i[:, None]) * sm_scale
        dq += tl.dot(ds.to(k.dtype), k)

    dq_ptrs = DQ + off_z * sdqz + off_h * sdqh + offs_m[:, None] * sdqm + offs_d[None, :] * sdqd
    tl.store(dq_ptrs, dq.to(DQ.dtype.element_ty), mask=m_valid[:, None])


def solution(q, k, v, o, do, L, delta, sm_scale, causal: bool=True):
    Z, H, N, D = q.shape
    dq = torch.empty_like(q)
    dk = torch.empty_like(k)
    dv = torch.empty_like(v)

    grid = lambda meta: (cdiv(N, meta['BLOCK']), Z * H)

    _bwd_dkdv[grid](
        q, k, v, do, dk, dv, L, delta, sm_scale, Z, H, N,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        do.stride(0), do.stride(1), do.stride(2), do.stride(3),
        dk.stride(0), dk.stride(1), dk.stride(2), dk.stride(3),
        dv.stride(0), dv.stride(1), dv.stride(2), dv.stride(3),
        L.stride(0), L.stride(1), L.stride(2),
        delta.stride(0), delta.stride(1), delta.stride(2),
        HEAD_DIM=D, CAUSAL=causal,
    )

    _bwd_dq[grid](
        q, k, v, do, dq, L, delta, sm_scale, Z, H, N,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        do.stride(0), do.stride(1), do.stride(2), do.stride(3),
        dq.stride(0), dq.stride(1), dq.stride(2), dq.stride(3),
        L.stride(0), L.stride(1), L.stride(2),
        delta.stride(0), delta.stride(1), delta.stride(2),
        HEAD_DIM=D, CAUSAL=causal,
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
