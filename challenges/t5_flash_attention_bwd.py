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

from gym import Challenge, register
from gym.tri import tl, jit, cdiv, require_triton


@jit
def _bwd_preprocess(O, DO, Delta, stride_z, stride_h, stride_n, stride_d,
                    H, N_CTX, HEAD_DIM: tl.constexpr, BLOCK_M: tl.constexpr):
    off_m = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    off_hz = tl.program_id(1)
    off_z = off_hz // H
    off_h = off_hz % H
    offs_d = tl.arange(0, HEAD_DIM)
    base = off_z * stride_z + off_h * stride_h
    mask = off_m[:, None] < N_CTX
    o = tl.load(O + base + off_m[:, None] * stride_n + offs_d[None, :] * stride_d,
                mask=mask, other=0.0).to(tl.float32)
    do = tl.load(DO + base + off_m[:, None] * stride_n + offs_d[None, :] * stride_d,
                 mask=mask, other=0.0).to(tl.float32)
    delta = tl.sum(o * do, axis=1)
    tl.store(Delta + off_hz * N_CTX + off_m, delta, mask=off_m < N_CTX)


@jit
def _bwd_dkdv(Q, K, V, DO, L, Delta, DK, DV, sm_scale,
              stride_z, stride_h, stride_n, stride_d, H, N_CTX,
              BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
              HEAD_DIM: tl.constexpr, CAUSAL: tl.constexpr):
    start_n = tl.program_id(0)
    off_hz = tl.program_id(1)
    off_z = off_hz // H
    off_h = off_hz % H
    base = off_z * stride_z + off_h * stride_h
    offs_n = start_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, HEAD_DIM)
    n_mask = offs_n < N_CTX

    k = tl.load(K + base + offs_n[:, None] * stride_n + offs_d[None, :] * stride_d,
                mask=n_mask[:, None], other=0.0)
    v = tl.load(V + base + offs_n[:, None] * stride_n + offs_d[None, :] * stride_d,
                mask=n_mask[:, None], other=0.0)
    dk = tl.zeros((BLOCK_N, HEAD_DIM), dtype=tl.float32)
    dv = tl.zeros((BLOCK_N, HEAD_DIM), dtype=tl.float32)

    lo = start_n * BLOCK_N if CAUSAL else 0
    lo = (lo // BLOCK_M) * BLOCK_M
    for start_m in range(lo, N_CTX, BLOCK_M):
        offs_m = start_m + tl.arange(0, BLOCK_M)
        m_mask = offs_m < N_CTX
        q = tl.load(Q + base + offs_m[:, None] * stride_n + offs_d[None, :] * stride_d,
                    mask=m_mask[:, None], other=0.0)
        do = tl.load(DO + base + offs_m[:, None] * stride_n + offs_d[None, :] * stride_d,
                     mask=m_mask[:, None], other=0.0)
        l_i = tl.load(L + off_hz * N_CTX + offs_m, mask=m_mask, other=0.0)
        delta = tl.load(Delta + off_hz * N_CTX + offs_m, mask=m_mask, other=0.0)

        qk = tl.dot(q, tl.trans(k)) * sm_scale               # (M, N)
        valid = m_mask[:, None] & n_mask[None, :]
        if CAUSAL:
            valid = valid & (offs_m[:, None] >= offs_n[None, :])
        qk = tl.where(valid, qk, -float("inf"))
        p = tl.exp(qk - l_i[:, None])                        # (M, N)
        dv += tl.dot(tl.trans(p.to(do.dtype)), do)           # (N, D)
        dp = tl.dot(do, tl.trans(v))                         # (M, N)
        ds = p * (dp - delta[:, None]) * sm_scale            # (M, N)
        ds = tl.where(valid, ds, 0.0)
        dk += tl.dot(tl.trans(ds.to(q.dtype)), q)            # (N, D)

    tl.store(DK + base + offs_n[:, None] * stride_n + offs_d[None, :] * stride_d,
             dk.to(DK.dtype.element_ty), mask=n_mask[:, None])
    tl.store(DV + base + offs_n[:, None] * stride_n + offs_d[None, :] * stride_d,
             dv.to(DV.dtype.element_ty), mask=n_mask[:, None])


@jit
def _bwd_dq(Q, K, V, DO, L, Delta, DQ, sm_scale,
            stride_z, stride_h, stride_n, stride_d, H, N_CTX,
            BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
            HEAD_DIM: tl.constexpr, CAUSAL: tl.constexpr):
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    off_z = off_hz // H
    off_h = off_hz % H
    base = off_z * stride_z + off_h * stride_h
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, HEAD_DIM)
    m_mask = offs_m < N_CTX

    q = tl.load(Q + base + offs_m[:, None] * stride_n + offs_d[None, :] * stride_d,
                mask=m_mask[:, None], other=0.0)
    do = tl.load(DO + base + offs_m[:, None] * stride_n + offs_d[None, :] * stride_d,
                 mask=m_mask[:, None], other=0.0)
    l_i = tl.load(L + off_hz * N_CTX + offs_m, mask=m_mask, other=0.0)
    delta = tl.load(Delta + off_hz * N_CTX + offs_m, mask=m_mask, other=0.0)
    dq = tl.zeros((BLOCK_M, HEAD_DIM), dtype=tl.float32)

    hi = (start_m + 1) * BLOCK_M if CAUSAL else N_CTX
    for start_n in range(0, hi, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)
        n_mask = offs_n < N_CTX
        k = tl.load(K + base + offs_n[:, None] * stride_n + offs_d[None, :] * stride_d,
                    mask=n_mask[:, None], other=0.0)
        v = tl.load(V + base + offs_n[:, None] * stride_n + offs_d[None, :] * stride_d,
                    mask=n_mask[:, None], other=0.0)
        qk = tl.dot(q, tl.trans(k)) * sm_scale
        valid = m_mask[:, None] & n_mask[None, :]
        if CAUSAL:
            valid = valid & (offs_m[:, None] >= offs_n[None, :])
        qk = tl.where(valid, qk, -float("inf"))
        p = tl.exp(qk - l_i[:, None])
        dp = tl.dot(do, tl.trans(v))
        ds = p * (dp - delta[:, None]) * sm_scale
        ds = tl.where(valid, ds, 0.0)
        dq += tl.dot(ds.to(k.dtype), k)

    tl.store(DQ + base + offs_m[:, None] * stride_n + offs_d[None, :] * stride_d,
             dq.to(DQ.dtype.element_ty), mask=m_mask[:, None])


def solution(q, k, v, o, do, L, delta, sm_scale, causal: bool = True):
    require_triton()
    q, k, v, do = q.contiguous(), k.contiguous(), v.contiguous(), do.contiguous()
    Z, H, N_CTX, HEAD_DIM = q.shape
    dq = torch.empty_like(q)
    dk = torch.empty_like(k)
    dv = torch.empty_like(v)
    BLOCK_M = BLOCK_N = 64
    sz, sh, sn, sd = q.stride(0), q.stride(1), q.stride(2), q.stride(3)
    grid_n = (cdiv(N_CTX, BLOCK_N), Z * H)
    _bwd_dkdv[grid_n](q, k, v, do, L, delta, dk, dv, sm_scale,
                      sz, sh, sn, sd, H, N_CTX,
                      BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, HEAD_DIM=HEAD_DIM, CAUSAL=causal,
                      num_warps=4, num_stages=2)
    grid_m = (cdiv(N_CTX, BLOCK_M), Z * H)
    _bwd_dq[grid_m](q, k, v, do, L, delta, dq, sm_scale,
                    sz, sh, sn, sd, H, N_CTX,
                    BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, HEAD_DIM=HEAD_DIM, CAUSAL=causal,
                    num_warps=4, num_stages=2)
    return dq, dk, dv


def reference(q, k, v, o, do, L, delta, sm_scale, causal: bool = True):
    qf = q.float().detach().requires_grad_(True)
    kf = k.float().detach().requires_grad_(True)
    vf = v.float().detach().requires_grad_(True)
    out = F.scaled_dot_product_attention(qf, kf, vf, is_causal=causal, scale=sm_scale)
    out.backward(do.float())
    return qf.grad.to(q.dtype), kf.grad.to(k.dtype), vf.grad.to(v.dtype)


def make_inputs(preset, device, dtype):
    shapes = {"small": (2, 4, 128, 64), "seq": (2, 8, 512, 64), "bench": (4, 16, 2048, 64)}
    Z, H, N, D = shapes[preset]
    sm_scale = 1.0 / math.sqrt(D)
    scale = 0.5
    q = torch.randn(Z, H, N, D, device=device, dtype=dtype) * scale
    k = torch.randn(Z, H, N, D, device=device, dtype=dtype) * scale
    v = torch.randn(Z, H, N, D, device=device, dtype=dtype) * scale
    do = torch.randn(Z, H, N, D, device=device, dtype=dtype) * scale
    # forward reference to produce o, L (natural logsumexp), and delta
    s = torch.matmul(q.float(), k.float().transpose(-1, -2)) * sm_scale
    causal_mask = torch.tril(torch.ones(N, N, device=device, dtype=torch.bool))
    s = s.masked_fill(~causal_mask, float("-inf"))
    L = torch.logsumexp(s, dim=-1)                       # (Z,H,N)
    p = torch.exp(s - L[..., None])
    o = torch.matmul(p, v.float()).to(dtype)
    delta = (o.float() * do.float()).sum(-1)             # (Z,H,N)
    return {"q": q, "k": k, "v": v, "o": o, "do": do,
            "L": L.contiguous().to(torch.float32), "delta": delta.contiguous().to(torch.float32),
            "sm_scale": sm_scale, "causal": True}


def _attn_bwd_flops(i):
    Z, H, N, D = i["q"].shape
    # backward is ~2.5x forward; forward causal ~ 2*2*Z*H*N*N*D*0.5
    return 2.5 * 2 * 2 * Z * H * N * N * D * 0.5


register(Challenge(
    id="t5.02", name="FlashAttention bwd (causal)", tier=5,
    description="dq/dk/dv by recomputing tiled softmax from stashed logsumexp; no N*N grads in HBM.",
    sources=[
        "@cloneofsimo — the backward pass is the hard, rarely-done part (fp8 FA3 bwd is job-worthy)",
        "@aryanvs_ — 75-80% of cuBLAS is weeks; the last 20% (training/backward kernels) is frontier",
        "@archiexzzz — differentiable recompute is what makes even double-backward possible",
    ],
    make_inputs=make_inputs, reference=reference, solution=solution,
    flops=_attn_bwd_flops,
    bytes=lambda i: 8 * i["q"].numel() * i["q"].element_size(),
    presets={"small": {}, "seq": {}, "bench": {}},
    dtypes=(torch.float16, torch.bfloat16),
    tol={torch.float16: (3e-2, 3e-2), torch.bfloat16: (5e-2, 5e-2)},
    grade_b=0.4, grade_a=0.7,
))
