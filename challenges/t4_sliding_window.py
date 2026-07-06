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

from gym import Challenge, register
from gym.tri import tl, jit, require_triton


@jit
def _swa_fwd_kernel(Q, K, V, sm_scale, Out,
                    stride_z, stride_h, stride_n, stride_d,
                    H, N_CTX, WINDOW,
                    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
                    HEAD_DIM: tl.constexpr):
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    off_z = off_hz // H
    off_h = off_hz % H
    base = off_z * stride_z + off_h * stride_h

    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, HEAD_DIM)
    m_mask = offs_m[:, None] < N_CTX
    q = tl.load(Q + base + offs_m[:, None] * stride_n + offs_d[None, :] * stride_d,
                mask=m_mask, other=0.0)

    m_i = tl.zeros((BLOCK_M,), dtype=tl.float32) - float("inf")
    l_i = tl.zeros((BLOCK_M,), dtype=tl.float32)
    acc = tl.zeros((BLOCK_M, HEAD_DIM), dtype=tl.float32)
    qk_scale = sm_scale * 1.44269504089

    # keys can only lie in [first_row - W + 1, last_row]: skip everything else
    qs = start_m * BLOCK_M
    lo = tl.maximum(qs - WINDOW + 1, 0)
    lo = (lo // BLOCK_N) * BLOCK_N
    hi = qs + BLOCK_M
    for start_n in range(lo, hi, BLOCK_N):
        cur_n = start_n + offs_n
        n_mask = cur_n < N_CTX
        k = tl.load(K + base + cur_n[None, :] * stride_n + offs_d[:, None] * stride_d,
                    mask=n_mask[None, :], other=0.0)
        qk = tl.dot(q, k) * qk_scale
        band = (offs_m[:, None] >= cur_n[None, :]) \
             & (offs_m[:, None] - cur_n[None, :] < WINDOW) \
             & n_mask[None, :]
        qk = tl.where(band, qk, -float("inf"))
        m_ij = tl.maximum(m_i, tl.max(qk, axis=1))
        # a row whose whole band lies outside this block has m_ij = -inf; using it in
        # exp2 gives exp2(-inf - -inf) = NaN. Guard the exponent with a safe max (the
        # row's p is 0 and acc stays 0 either way); keep the true -inf in m_i so a later
        # block that DOES contain valid keys rescales correctly.
        m_safe = tl.where(m_ij == -float("inf"), 0.0, m_ij)
        p = tl.exp2(qk - m_safe[:, None])
        alpha = tl.exp2(m_i - m_safe)
        l_i = l_i * alpha + tl.sum(p, axis=1)
        acc = acc * alpha[:, None]
        v = tl.load(V + base + cur_n[:, None] * stride_n + offs_d[None, :] * stride_d,
                    mask=n_mask[:, None], other=0.0)
        acc += tl.dot(p.to(v.dtype), v)
        m_i = m_ij

    l_i = tl.where(l_i == 0.0, 1.0, l_i)
    acc = acc / l_i[:, None]
    tl.store(Out + base + offs_m[:, None] * stride_n + offs_d[None, :] * stride_d,
             acc.to(Out.dtype.element_ty), mask=m_mask)


def solution(q, k, v, window: int, mask=None):
    require_triton()
    q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
    Z, H, N_CTX, HEAD_DIM = q.shape
    sm_scale = 1.0 / math.sqrt(HEAD_DIM)
    o = torch.empty_like(q)
    BLOCK_M = BLOCK_N = 64
    grid = ((N_CTX + BLOCK_M - 1) // BLOCK_M, Z * H)
    _swa_fwd_kernel[grid](q, k, v, sm_scale, o,
                          q.stride(0), q.stride(1), q.stride(2), q.stride(3),
                          H, N_CTX, window,
                          BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, HEAD_DIM=HEAD_DIM,
                          num_warps=4, num_stages=2)
    return o


def reference(q, k, v, window: int, mask=None):
    if mask is None:
        N = q.shape[2]
        i = torch.arange(N, device=q.device)
        mask = (i[:, None] >= i[None, :]) & (i[:, None] - i[None, :] < window)
    return F.scaled_dot_product_attention(q, k, v, attn_mask=mask)


def make_inputs(preset, device, dtype):
    shapes = {"small": (2, 4, 128, 64, 32), "seq": (2, 8, 1024, 64, 128),
              "bench": (2, 8, 4096, 64, 256)}
    Z, H, N, D, W = shapes[preset]
    s = 0.5
    i = torch.arange(N, device=device)
    mask = (i[:, None] >= i[None, :]) & (i[:, None] - i[None, :] < W)
    return {"q": torch.randn(Z, H, N, D, device=device, dtype=dtype) * s,
            "k": torch.randn(Z, H, N, D, device=device, dtype=dtype) * s,
            "v": torch.randn(Z, H, N, D, device=device, dtype=dtype) * s,
            "window": W, "mask": mask}


def _swa_flops(i):
    Z, H, N, D = i["q"].shape
    W = i["window"]
    pairs = (N - W) * W + W * (W + 1) // 2      # per (z, h): banded qk pairs
    return 2 * 2 * Z * H * pairs * D


register(Challenge(
    id="t4.05", name="sliding-window attention", tier=4,
    description="Banded causal attention; out-of-window K/V blocks never loaded — O(N*W) work.",
    sources=[
        "@karpathy — FA3 window_size enables alternating local/global layers",
        "@cHHillee — local attention is remediation #1 for linear-in-context KV memory",
        "@hamzaelshafie (gpt-oss) — banded/sliding-window attention in alternating layers",
    ],
    make_inputs=make_inputs, reference=reference, solution=solution,
    flops=_swa_flops,
    bytes=lambda i: 4 * i["q"].numel() * i["q"].element_size(),
    presets={"small": {}, "seq": {}, "bench": {}},
    dtypes=(torch.float16, torch.bfloat16),
    tol={torch.float16: (2e-2, 2e-2), torch.bfloat16: (3e-2, 3e-2)},
    grade_b=2.0, grade_a=4.0,
))
