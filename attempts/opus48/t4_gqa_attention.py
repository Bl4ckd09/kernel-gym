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

@triton.jit
def _gqa_flash_kernel(q_ptr, k_ptr, v_ptr, o_ptr,
                      stride_qz, stride_qh, stride_qn, stride_qd,
                      stride_kz, stride_kh, stride_kn, stride_kd,
                      stride_vz, stride_vh, stride_vn, stride_vd,
                      stride_oz, stride_oh, stride_on, stride_od,
                      Hq, N, G, scale,
                      causal: tl.constexpr,
                      BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
                      BLOCK_D: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_zh = tl.program_id(1)
    z = pid_zh // Hq
    hq = pid_zh % Hq
    hkv = hq // G  # GQA: query head hq reads KV head hq // G

    q_base = z.to(tl.int64) * stride_qz + hq * stride_qh
    k_base = z.to(tl.int64) * stride_kz + hkv * stride_kh
    v_base = z.to(tl.int64) * stride_vz + hkv * stride_vh
    o_base = z.to(tl.int64) * stride_oz + hq * stride_oh

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_D)
    mask_m = offs_m < N

    q_ptrs = q_ptr + q_base + offs_m[:, None] * stride_qn + offs_d[None, :] * stride_qd
    q = tl.load(q_ptrs, mask=mask_m[:, None], other=0.0)

    m_i = tl.full((BLOCK_M,), -float('inf'), dtype=tl.float32)
    l_i = tl.zeros((BLOCK_M,), dtype=tl.float32)
    acc = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)

    if causal:
        hi = tl.minimum((pid_m + 1) * BLOCK_M, N)
    else:
        hi = N

    for start_n in range(0, hi, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)
        mask_n = offs_n < N
        k_ptrs = k_ptr + k_base + offs_d[:, None] * stride_kd + offs_n[None, :] * stride_kn
        k = tl.load(k_ptrs, mask=mask_n[None, :], other=0.0)
        qk = tl.dot(q, k).to(tl.float32) * scale
        qk = tl.where(mask_n[None, :], qk, -float('inf'))
        if causal:
            qk = tl.where(offs_m[:, None] >= offs_n[None, :], qk, -float('inf'))
        m_ij = tl.maximum(m_i, tl.max(qk, axis=1))
        p = tl.exp(qk - m_ij[:, None])
        alpha = tl.exp(m_i - m_ij)
        l_i = l_i * alpha + tl.sum(p, axis=1)
        v_ptrs = v_ptr + v_base + offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vd
        v = tl.load(v_ptrs, mask=mask_n[:, None], other=0.0)
        acc = acc * alpha[:, None] + tl.dot(p.to(q.dtype), v)
        m_i = m_ij

    acc = acc / l_i[:, None]
    o_ptrs = o_ptr + o_base + offs_m[:, None] * stride_on + offs_d[None, :] * stride_od
    tl.store(o_ptrs, acc.to(o_ptr.dtype.element_ty), mask=mask_m[:, None])


def solution(q, k, v, causal: bool=True):
    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()
    Z, Hq, N, D = q.shape
    Hkv = k.shape[1]
    G = Hq // Hkv
    o = torch.empty_like(q)
    scale = 1.0 / math.sqrt(D)
    BLOCK_M = 64
    BLOCK_N = 64
    BLOCK_D = D  # 64, power of 2
    grid = (triton.cdiv(N, BLOCK_M), Z * Hq)
    _gqa_flash_kernel[grid](
        q, k, v, o,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        o.stride(0), o.stride(1), o.stride(2), o.stride(3),
        Hq, N, G, scale, causal,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_D=BLOCK_D,
        num_warps=4, num_stages=2,
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
