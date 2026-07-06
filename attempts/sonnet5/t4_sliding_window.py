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


@autotune(
    configs=[
        Config({'BLOCK_M': 64, 'BLOCK_N': 64}, num_warps=4, num_stages=2),
        Config({'BLOCK_M': 128, 'BLOCK_N': 64}, num_warps=8, num_stages=2),
    ],
    key=['N', 'D', 'window'],
)
@jit
def _sliding_window_attn_fwd_kernel(
    q_ptr, k_ptr, v_ptr, o_ptr,
    stride_qz, stride_qh, stride_qn, stride_qd,
    stride_kz, stride_kh, stride_kn, stride_kd,
    stride_vz, stride_vh, stride_vn, stride_vd,
    stride_oz, stride_oh, stride_on, stride_od,
    H, N, D, window,
    scale,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_D: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_zh = tl.program_id(1)
    z = pid_zh // H
    h = pid_zh % H

    q_base = q_ptr + z * stride_qz + h * stride_qh
    k_base = k_ptr + z * stride_kz + h * stride_kh
    v_base = v_ptr + z * stride_vz + h * stride_vh
    o_base = o_ptr + z * stride_oz + h * stride_oh

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_D)

    q_mask = (offs_m[:, None] < N) & (offs_d[None, :] < D)
    q_ptrs = q_base + offs_m[:, None] * stride_qn + offs_d[None, :] * stride_qd
    q = tl.load(q_ptrs, mask=q_mask, other=0.0)

    m_i = tl.full((BLOCK_M,), value=float('-inf'), dtype=tl.float32)
    l_i = tl.zeros((BLOCK_M,), dtype=tl.float32)
    acc = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)

    qstart = pid_m * BLOCK_M
    qend = qstart + BLOCK_M

    lo = tl.maximum(qstart - window + 1, 0)
    lo = (lo // BLOCK_N) * BLOCK_N
    hi = qend

    for start_n in range(lo, hi, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)

        k_mask = (offs_n[:, None] < N) & (offs_d[None, :] < D)
        k_ptrs = k_base + offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kd
        k = tl.load(k_ptrs, mask=k_mask, other=0.0)

        qk = tl.dot(q, tl.trans(k)) * scale

        diff = offs_m[:, None] - offs_n[None, :]
        valid = (offs_m[:, None] < N) & (offs_n[None, :] < N) & (diff >= 0) & (diff < window)
        qk = tl.where(valid, qk, float('-inf'))

        row_max = tl.max(qk, axis=1)
        m_ij = tl.maximum(m_i, row_max)
        # NaN guard: if this row has seen no valid key yet (m_ij == -inf), the
        # subtraction below would be (-inf) - (-inf) = NaN. Replace the
        # subtrahend with a finite stand-in in that case; since qk is also
        # all -inf for such a row, exp(-inf - 0) = 0, which is the correct
        # contribution (no keys in-window yet).
        m_ij_safe = tl.where(m_ij == float('-inf'), 0.0, m_ij)

        p = tl.exp(qk - m_ij_safe[:, None])
        l_ij = tl.sum(p, axis=1)

        alpha = tl.exp(m_i - m_ij_safe)
        l_i = l_i * alpha + l_ij
        acc = acc * alpha[:, None]

        v_mask = (offs_n[:, None] < N) & (offs_d[None, :] < D)
        v_ptrs = v_base + offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vd
        v = tl.load(v_ptrs, mask=v_mask, other=0.0)

        acc += tl.dot(p.to(v.dtype), v)

        m_i = m_ij

    l_i_safe = tl.where(l_i == 0.0, 1.0, l_i)
    acc = acc / l_i_safe[:, None]

    o_mask = (offs_m[:, None] < N) & (offs_d[None, :] < D)
    o_ptrs = o_base + offs_m[:, None] * stride_on + offs_d[None, :] * stride_od
    tl.store(o_ptrs, acc.to(o_ptr.dtype.element_ty), mask=o_mask)


def solution(q, k, v, window: int, mask=None):
    """YOUR KERNEL HERE — see rules at top of file."""
    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()

    Z, H, N, D = q.shape

    o = torch.empty_like(q)
    scale = 1.0 / math.sqrt(D)

    BLOCK_D = triton.next_power_of_2(D)

    grid = lambda meta: (cdiv(N, meta['BLOCK_M']), Z * H)

    _sliding_window_attn_fwd_kernel[grid](
        q, k, v, o,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        o.stride(0), o.stride(1), o.stride(2), o.stride(3),
        H, N, D, window,
        scale,
        BLOCK_D=BLOCK_D,
    )

    return o

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
