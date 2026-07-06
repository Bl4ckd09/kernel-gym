"""t4.03 — GQA FlashAttention forward (causal): G query heads share one KV head.

Grouped-query attention is why modern decode is affordable: with H_q query heads but only
H_kv KV heads (group size G = H_q/H_kv), the KV cache and its bandwidth shrink by G with
almost no quality loss. The kernel is t5.01's online-softmax flash forward with one twist:
program (z, h_q) reads K/V from head h_q // G. Get the indexing wrong and heads silently
attend to the wrong memory — correctness is checked against an expanded-KV reference.

The eager baseline must materialize the expanded K/V (repeat_interleave) before calling
SDPA, paying a G-times-larger KV copy; the kernel reads the compact K/V directly. That
copy is the win being graded.

Notes lineage: @rohanpaul_ai — MQA/GQA cut KV cache and bandwidth; @GoSailGlobal (CS336)
— GQA cuts decode cost ~80%, naive MHA collapses arithmetic intensity to ~1/h during
decode; @TheAhmadOsman — implement GQA and sweep #KV groups.
"""

import math

import torch
import torch.nn.functional as F

from gym import Challenge, register
from gym.tri import tl, jit, require_triton


@jit
def _gqa_fwd_kernel(Q, K, V, sm_scale, Out,
                    stride_qz, stride_qh, stride_qm, stride_qk,
                    stride_kz, stride_kh, stride_kn, stride_kk,
                    stride_vz, stride_vh, stride_vn, stride_vk,
                    stride_oz, stride_oh, stride_om, stride_ok,
                    H, N_CTX, GROUP: tl.constexpr,
                    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
                    HEAD_DIM: tl.constexpr, CAUSAL: tl.constexpr):
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    off_z = off_hz // H
    off_h = off_hz % H
    off_h_kv = off_h // GROUP          # the whole challenge, in one line
    q_base = Q + off_z * stride_qz + off_h * stride_qh
    k_base = K + off_z * stride_kz + off_h_kv * stride_kh
    v_base = V + off_z * stride_vz + off_h_kv * stride_vh

    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, HEAD_DIM)

    q_ptrs = q_base + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qk
    m_mask = offs_m[:, None] < N_CTX
    q = tl.load(q_ptrs, mask=m_mask, other=0.0)

    m_i = tl.zeros((BLOCK_M,), dtype=tl.float32) - float("inf")
    l_i = tl.zeros((BLOCK_M,), dtype=tl.float32)
    acc = tl.zeros((BLOCK_M, HEAD_DIM), dtype=tl.float32)
    qk_scale = sm_scale * 1.44269504089

    hi = (start_m + 1) * BLOCK_M if CAUSAL else N_CTX
    for start_n in range(0, hi, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        cur_n = start_n + offs_n
        n_mask = cur_n < N_CTX
        k = tl.load(k_base + cur_n[None, :] * stride_kn + offs_d[:, None] * stride_kk,
                    mask=n_mask[None, :], other=0.0)
        qk = tl.dot(q, k) * qk_scale
        qk = tl.where(n_mask[None, :], qk, -float("inf"))
        if CAUSAL:
            qk = tl.where(offs_m[:, None] >= cur_n[None, :], qk, -float("inf"))
        m_ij = tl.maximum(m_i, tl.max(qk, axis=1))
        p = tl.exp2(qk - m_ij[:, None])
        alpha = tl.exp2(m_i - m_ij)
        l_i = l_i * alpha + tl.sum(p, axis=1)
        acc = acc * alpha[:, None]
        v = tl.load(v_base + cur_n[:, None] * stride_vn + offs_d[None, :] * stride_vk,
                    mask=n_mask[:, None], other=0.0)
        acc += tl.dot(p.to(v.dtype), v)
        m_i = m_ij

    l_i = tl.where(l_i == 0.0, 1.0, l_i)
    acc = acc / l_i[:, None]
    o_ptrs = Out + off_z * stride_oz + off_h * stride_oh \
        + offs_m[:, None] * stride_om + offs_d[None, :] * stride_ok
    tl.store(o_ptrs, acc.to(Out.dtype.element_ty), mask=m_mask)


def solution(q, k, v, causal: bool = True):
    require_triton()
    q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
    Z, Hq, N_CTX, HEAD_DIM = q.shape
    Hkv = k.shape[1]
    GROUP = Hq // Hkv
    sm_scale = 1.0 / math.sqrt(HEAD_DIM)
    o = torch.empty_like(q)
    BLOCK_M, BLOCK_N = 64, 64
    grid = ((N_CTX + BLOCK_M - 1) // BLOCK_M, Z * Hq)
    _gqa_fwd_kernel[grid](
        q, k, v, sm_scale, o,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        o.stride(0), o.stride(1), o.stride(2), o.stride(3),
        Hq, N_CTX, GROUP=GROUP,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, HEAD_DIM=HEAD_DIM, CAUSAL=causal,
        num_warps=4, num_stages=2,
    )
    return o


def reference(q, k, v, causal: bool = True):
    G = q.shape[1] // k.shape[1]
    k_exp = k.repeat_interleave(G, dim=1)
    v_exp = v.repeat_interleave(G, dim=1)
    return F.scaled_dot_product_attention(q, k_exp, v_exp, is_causal=causal)


def make_inputs(preset, device, dtype):
    shapes = {"small": (2, 8, 2, 128, 64), "seq": (2, 8, 2, 1024, 64),
              "bench": (4, 32, 8, 4096, 64)}
    Z, Hq, Hkv, N, D = shapes[preset]
    s = 0.5
    return {"q": torch.randn(Z, Hq, N, D, device=device, dtype=dtype) * s,
            "k": torch.randn(Z, Hkv, N, D, device=device, dtype=dtype) * s,
            "v": torch.randn(Z, Hkv, N, D, device=device, dtype=dtype) * s,
            "causal": True}


def _gqa_flops(i):
    Z, Hq, N, D = i["q"].shape
    return 2 * 2 * Z * Hq * N * N * D * 0.5


register(Challenge(
    id="t4.03", name="GQA FlashAttention fwd", tier=4,
    description="Flash forward where G query heads read one shared KV head; compact K/V, no expansion.",
    sources=[
        "@rohanpaul_ai — MQA/GQA cut KV cache and its bandwidth",
        "@GoSailGlobal (CS336) — GQA cuts decode cost ~80%; MHA decode AI collapses to ~1/h",
        "@TheAhmadOsman — implement GQA, sweep #KV groups",
    ],
    make_inputs=make_inputs, reference=reference, solution=solution,
    flops=_gqa_flops,
    bytes=lambda i: (2 * i["q"].numel() + 2 * i["k"].numel()) * i["q"].element_size(),
    presets={"small": {}, "seq": {}, "bench": {}},
    dtypes=(torch.float16, torch.bfloat16),
    tol={torch.float16: (2e-2, 2e-2), torch.bfloat16: (3e-2, 3e-2)},
    # baseline pays the G-times KV expansion; beating it at all means compact-KV wins
    grade_b=0.85, grade_a=1.0,
))
