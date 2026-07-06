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

from gym import Challenge, register
from gym.tri import tl, jit, require_triton


@jit
def _flash_fwd_kernel(Q, K, V, sm_scale, Out, L,
                      stride_qz, stride_qh, stride_qm, stride_qk,
                      stride_kz, stride_kh, stride_kn, stride_kk,
                      stride_vz, stride_vh, stride_vn, stride_vk,
                      stride_oz, stride_oh, stride_om, stride_ok,
                      Z, H, N_CTX,
                      BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
                      HEAD_DIM: tl.constexpr, CAUSAL: tl.constexpr):
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    off_z = off_hz // H
    off_h = off_hz % H
    q_base = Q + off_z * stride_qz + off_h * stride_qh
    k_base = K + off_z * stride_kz + off_h * stride_kh
    v_base = V + off_z * stride_vz + off_h * stride_vh

    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, HEAD_DIM)

    q_ptrs = q_base + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qk
    m_mask = offs_m[:, None] < N_CTX
    q = tl.load(q_ptrs, mask=m_mask, other=0.0)

    m_i = tl.zeros((BLOCK_M,), dtype=tl.float32) - float("inf")
    l_i = tl.zeros((BLOCK_M,), dtype=tl.float32)
    acc = tl.zeros((BLOCK_M, HEAD_DIM), dtype=tl.float32)
    qk_scale = sm_scale * 1.44269504089  # scale into exp2 domain

    hi = (start_m + 1) * BLOCK_M if CAUSAL else N_CTX
    for start_n in range(0, hi, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        cur_n = start_n + offs_n
        n_mask = cur_n < N_CTX
        k_ptrs = k_base + cur_n[None, :] * stride_kn + offs_d[:, None] * stride_kk
        v_ptrs = v_base + cur_n[:, None] * stride_vn + offs_d[None, :] * stride_vk
        k = tl.load(k_ptrs, mask=n_mask[None, :], other=0.0)
        qk = tl.dot(q, k) * qk_scale
        # mask out-of-range and (if causal) future keys
        qk = tl.where(n_mask[None, :], qk, -float("inf"))
        if CAUSAL:
            causal = offs_m[:, None] >= cur_n[None, :]
            qk = tl.where(causal, qk, -float("inf"))
        m_ij = tl.maximum(m_i, tl.max(qk, axis=1))
        p = tl.exp2(qk - m_ij[:, None])
        alpha = tl.exp2(m_i - m_ij)
        l_i = l_i * alpha + tl.sum(p, axis=1)
        acc = acc * alpha[:, None]
        v = tl.load(v_ptrs, mask=n_mask[:, None], other=0.0)
        acc += tl.dot(p.to(v.dtype), v)
        m_i = m_ij

    l_i = tl.where(l_i == 0.0, 1.0, l_i)  # fully-masked rows (shouldn't happen for causal)
    acc = acc / l_i[:, None]
    o_ptrs = Out + off_z * stride_oz + off_h * stride_oh \
        + offs_m[:, None] * stride_om + offs_d[None, :] * stride_ok
    tl.store(o_ptrs, acc.to(Out.dtype.element_ty), mask=m_mask)
    if L is not None:
        l_ptrs = L + off_hz * N_CTX + offs_m
        tl.store(l_ptrs, m_i + tl.log2(l_i), mask=offs_m < N_CTX)


def solution(q, k, v, causal: bool = True):
    require_triton()
    q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
    Z, H, N_CTX, HEAD_DIM = q.shape
    sm_scale = 1.0 / math.sqrt(HEAD_DIM)
    o = torch.empty_like(q)
    BLOCK_M, BLOCK_N = 64, 64
    grid = ((N_CTX + BLOCK_M - 1) // BLOCK_M, Z * H)
    _flash_fwd_kernel[grid](
        q, k, v, sm_scale, o, None,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        o.stride(0), o.stride(1), o.stride(2), o.stride(3),
        Z, H, N_CTX,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, HEAD_DIM=HEAD_DIM, CAUSAL=causal,
        num_warps=4, num_stages=2,
    )
    return o


def reference(q, k, v, causal: bool = True):
    return F.scaled_dot_product_attention(q, k, v, is_causal=causal)


def make_inputs(preset, device, dtype):
    shapes = {"small": (2, 4, 128, 64), "seq": (2, 8, 1024, 64), "bench": (4, 16, 4096, 64)}
    Z, H, N, D = shapes[preset]
    scale = 0.5
    return {"q": torch.randn(Z, H, N, D, device=device, dtype=dtype) * scale,
            "k": torch.randn(Z, H, N, D, device=device, dtype=dtype) * scale,
            "v": torch.randn(Z, H, N, D, device=device, dtype=dtype) * scale,
            "causal": True}


def _attn_flops(i):
    Z, H, N, D = i["q"].shape
    # 2 matmuls (QK^T and PV), each 2*N*N*D, halved for causal
    return 2 * 2 * Z * H * N * N * D * 0.5


register(Challenge(
    id="t5.01", name="FlashAttention fwd (causal)", tier=5,
    description="Online-softmax tiled attention; running max/denom rescale; no N*N scores in HBM.",
    sources=[
        "@tri_dao — online/streaming softmax is the heart of FlashAttention",
        "@pranay5255 — FA2's 2-3x over FA1 is the Q-outer / KV-inner loop order",
        "@cHHillee — same softmax math as t2.01, generalized to tiles that don't fit SRAM",
    ],
    make_inputs=make_inputs, reference=reference, solution=solution,
    flops=_attn_flops,
    bytes=lambda i: 4 * i["q"].numel() * i["q"].element_size(),
    presets={"small": {}, "seq": {}, "bench": {}},
    dtypes=(torch.float16, torch.bfloat16),
    tol={torch.float16: (2e-2, 2e-2), torch.bfloat16: (3e-2, 3e-2)},
    grade_b=0.5, grade_a=0.85,
))
