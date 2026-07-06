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

@triton.jit
def _swa_fwd_kernel(
    Q, K, V, Out, sm_scale,
    stride_qz, stride_qh, stride_qm, stride_qd,
    stride_kz, stride_kh, stride_kn, stride_kd,
    stride_vz, stride_vh, stride_vn, stride_vd,
    stride_oz, stride_oh, stride_om, stride_od,
    N, W, H,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_D: tl.constexpr,
    IP: tl.constexpr,
):
    start_m = tl.program_id(0)
    off_zh = tl.program_id(1)
    off_z = off_zh // H
    off_h = off_zh % H

    q_base = Q + off_z * stride_qz + off_h * stride_qh
    k_base = K + off_z * stride_kz + off_h * stride_kh
    v_base = V + off_z * stride_vz + off_h * stride_vh
    o_base = Out + off_z * stride_oz + off_h * stride_oh

    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_D)
    m_valid = offs_m < N

    q_ptrs = q_base + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qd
    q = tl.load(q_ptrs, mask=m_valid[:, None], other=0.0)

    qk_scale = sm_scale * 1.4426950408889634  # 1/ln2, fold into exp2

    m_i = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)

    # band of key indices needed by this query block -> COMPUTE SKIPPING
    lo = start_m * BLOCK_M - W + 1
    lo = tl.maximum(lo, 0)
    lo = (lo // BLOCK_N) * BLOCK_N
    hi = tl.minimum(start_m * BLOCK_M + BLOCK_M, N)

    for start_n in range(lo, hi, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        offs_n = start_n + tl.arange(0, BLOCK_N)
        n_valid = offs_n < N
        # K loaded as (BLOCK_D, BLOCK_N) so q @ k -> (BLOCK_M, BLOCK_N)
        k_ptrs = k_base + offs_d[:, None] * stride_kd + offs_n[None, :] * stride_kn
        k = tl.load(k_ptrs, mask=n_valid[None, :], other=0.0)
        qk = tl.dot(q, k, input_precision=IP) * qk_scale
        diff = offs_m[:, None] - offs_n[None, :]
        band = (diff >= 0) & (diff < W) & n_valid[None, :]
        qk = tl.where(band, qk, float("-inf"))
        m_ij = tl.max(qk, 1)
        m_new = tl.maximum(m_i, m_ij)
        # NaN trap guard: rows whose whole band is outside this block have
        # m_new = -inf; substitute a finite 0.0 in the exp2 exponents only.
        m_safe = tl.where(m_new == float("-inf"), 0.0, m_new)
        alpha = tl.exp2(m_i - m_safe)
        p = tl.exp2(qk - m_safe[:, None])
        l_i = l_i * alpha + tl.sum(p, 1)
        acc = acc * alpha[:, None]
        v_ptrs = v_base + offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vd
        v = tl.load(v_ptrs, mask=n_valid[:, None], other=0.0)
        acc += tl.dot(p.to(v.dtype), v, input_precision=IP)
        m_i = m_new  # keep the TRUE running max (may be -inf)

    l_safe = tl.where(l_i == 0.0, 1.0, l_i)
    acc = acc / l_safe[:, None]
    o_ptrs = o_base + offs_m[:, None] * stride_om + offs_d[None, :] * stride_od
    tl.store(o_ptrs, acc.to(Out.dtype.element_ty), mask=m_valid[:, None])


def solution(q, k, v, window: int, mask=None):
    Z, H, N, D = q.shape
    q = q.contiguous(); k = k.contiguous(); v = v.contiguous()
    out = torch.empty_like(q)
    sm_scale = 1.0 / math.sqrt(D)
    ip = "ieee" if q.dtype == torch.float32 else "tf32"
    BLOCK_M = 64
    BLOCK_N = 64
    grid = (triton.cdiv(N, BLOCK_M), Z * H)
    _swa_fwd_kernel[grid](
        q, k, v, out, sm_scale,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        N, window, H,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_D=D, IP=ip,
        num_warps=4, num_stages=3,
    )
    return out

def reference(q, k, v, window: int, mask=None):
    if mask is None:
        N = q.shape[2]
        i = torch.arange(N, device=q.device)
        mask = (i[:, None] >= i[None, :]) & (i[:, None] - i[None, :] < window)
    return F.scaled_dot_product_attention(q, k, v, attn_mask=mask)

def make_inputs(preset, device, dtype):
    shapes = {'small': (2, 4, 128, 64, 32), 'seq': (2, 8, 1024, 64, 128), 'bench': (2, 8, 4096, 64, 256)}
    Z, H, N, D, W = shapes[preset]
    s = 0.5
    i = torch.arange(N, device=device)
    mask = (i[:, None] >= i[None, :]) & (i[:, None] - i[None, :] < W)
    return {'q': torch.randn(Z, H, N, D, device=device, dtype=dtype) * s, 'k': torch.randn(Z, H, N, D, device=device, dtype=dtype) * s, 'v': torch.randn(Z, H, N, D, device=device, dtype=dtype) * s, 'window': W, 'mask': mask}

def _swa_flops(i):
    Z, H, N, D = i['q'].shape
    W = i['window']
    pairs = (N - W) * W + W * (W + 1) // 2
    return 2 * 2 * Z * H * pairs * D
