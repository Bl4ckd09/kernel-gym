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


@autotune(
    configs=[
        Config({'BLOCK_M': 64, 'BLOCK_N': 64}, num_warps=4, num_stages=2),
        Config({'BLOCK_M': 128, 'BLOCK_N': 64}, num_warps=8, num_stages=2),
        Config({'BLOCK_M': 64, 'BLOCK_N': 128}, num_warps=8, num_stages=2),
    ],
    key=['N', 'D'],
)
@jit
def _gqa_attn_fwd_kernel(
    q_ptr, k_ptr, v_ptr, o_ptr,
    stride_qz, stride_qh, stride_qn, stride_qd,
    stride_kz, stride_kh, stride_kn, stride_kd,
    stride_vz, stride_vh, stride_vn, stride_vd,
    stride_oz, stride_oh, stride_on, stride_od,
    Hq, N, D,
    G: tl.constexpr,
    scale,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_D: tl.constexpr,
    CAUSAL: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_zh = tl.program_id(1)
    z = pid_zh // Hq
    h = pid_zh % Hq
    kv_h = h // G

    q_base = q_ptr + z * stride_qz + h * stride_qh
    k_base = k_ptr + z * stride_kz + kv_h * stride_kh
    v_base = v_ptr + z * stride_vz + kv_h * stride_vh
    o_base = o_ptr + z * stride_oz + h * stride_oh

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_D)

    q_mask = (offs_m[:, None] < N) & (offs_d[None, :] < D)
    q_ptrs = q_base + offs_m[:, None] * stride_qn + offs_d[None, :] * stride_qd
    q = tl.load(q_ptrs, mask=q_mask, other=0.0)

    m_i = tl.full((BLOCK_M,), value=float('-inf'), dtype=tl.float32)
    l_i = tl.zeros((BLOCK_M,), dtype=tl.float32)
    acc = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)

    if CAUSAL:
        hi = (pid_m + 1) * BLOCK_M
    else:
        hi = N

    for start_n in range(0, hi, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)

        k_mask = (offs_n[:, None] < N) & (offs_d[None, :] < D)
        k_ptrs = k_base + offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kd
        k = tl.load(k_ptrs, mask=k_mask, other=0.0)

        qk = tl.dot(q, tl.trans(k)) * scale

        valid = (offs_m[:, None] < N) & (offs_n[None, :] < N)
        if CAUSAL:
            valid = valid & (offs_m[:, None] >= offs_n[None, :])
        qk = tl.where(valid, qk, float('-inf'))

        m_ij = tl.maximum(m_i, tl.max(qk, axis=1))
        p = tl.exp(qk - m_ij[:, None])
        l_ij = tl.sum(p, axis=1)

        alpha = tl.exp(m_i - m_ij)
        l_i = l_i * alpha + l_ij
        acc = acc * alpha[:, None]

        v_mask = (offs_n[:, None] < N) & (offs_d[None, :] < D)
        v_ptrs = v_base + offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vd
        v = tl.load(v_ptrs, mask=v_mask, other=0.0)

        acc += tl.dot(p.to(v.dtype), v)

        m_i = m_ij

    acc = acc / l_i[:, None]

    o_mask = (offs_m[:, None] < N) & (offs_d[None, :] < D)
    o_ptrs = o_base + offs_m[:, None] * stride_on + offs_d[None, :] * stride_od
    tl.store(o_ptrs, acc.to(o_ptr.dtype.element_ty), mask=o_mask)


def solution(q, k, v, causal: bool=True):
    """YOUR KERNEL HERE — see rules at top of file."""
    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()

    Z, Hq, N, D = q.shape
    _, Hkv, Nk, Dk = k.shape
    G = Hq // Hkv

    o = torch.empty_like(q)
    scale = 1.0 / math.sqrt(D)

    BLOCK_D = triton.next_power_of_2(D)

    grid = lambda meta: (cdiv(N, meta['BLOCK_M']), Z * Hq)

    _gqa_attn_fwd_kernel[grid](
        q, k, v, o,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        o.stride(0), o.stride(1), o.stride(2), o.stride(3),
        Hq, N, D,
        G,
        scale,
        BLOCK_D=BLOCK_D,
        CAUSAL=causal,
    )

    return o

def reference(q, k, v, causal: bool=True):
    G = q.shape[1] // k.shape[1]
    k_exp = k.repeat_interleave(G, dim=1)
    v_exp = v.repeat_interleave(G, dim=1)
    return F.scaled_dot_product_attention(q, k_exp, v_exp, is_causal=causal)

def make_inputs(preset, device, dtype):
    shapes = {'small': (2, 8, 2, 128, 64), 'seq': (2, 8, 2, 1024, 64), 'bench': (4, 32, 8, 4096, 64)}
    Z, Hq, Hkv, N, D = shapes[preset]
    s = 0.5
    return {'q': torch.randn(Z, Hq, N, D, device=device, dtype=dtype) * s, 'k': torch.randn(Z, Hkv, N, D, device=device, dtype=dtype) * s, 'v': torch.randn(Z, Hkv, N, D, device=device, dtype=dtype) * s, 'causal': True}

def _gqa_flops(i):
    Z, Hq, N, D = i['q'].shape
    return 2 * 2 * Z * Hq * N * N * D * 0.5
