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

"""t4.04 — W4A16 dequant matmul: int4 weights, fp16 activations, decode-shaped.

The Marlin / Kimi-K2 inference path. Weights live in HBM as packed int4 — two values per
uint8, plus one fp16 scale per 64-element group along K — and are unpacked to fp16
in-register, never materialized. out = x @ W^T with W[n,k] = (nibble - 8) * scale[n, k//64].

Why it wins: at decode (small M) the matmul is bandwidth-bound on the WEIGHTS. fp16
weights cost 2 bytes each; packed int4 costs 0.5 + a sliver of scale — ~4x less traffic,
so the roofline says up to ~4x faster than the fp16 matmul the baseline runs. The
challenge is the in-register unpack: nibble extraction, interleave back into K order,
dequant, straight into tl.dot, fp32 accumulate.

The baseline gets the PRE-DEQUANTIZED fp16 weights for free (they're in the inputs as
`w_fp16` — your kernel should ignore them); it models a server that kept weights in fp16.

Notes lineage: @jeremyphoward/@Yuchenj_UW/@doodlestein — Kimi K2 native INT4 (1x32 block
scale) on Marlin kernels; int4 halves decode latency, advantage grows with context;
@maharshii — FP4/INT4 bit-packing: two nibbles per uint8, unpack with & 0xF and >> 4;
@elliotarledge — w4a16 GEMM written cleanly is a KernelBench-Hard problem.
"""
import torch
GROUP = 64

@triton.jit
def _w4a16_kernel(x_ptr, wp_ptr, s_ptr, o_ptr,
                  M, N, K,
                  stride_xm, stride_xk,
                  stride_wn, stride_wk,
                  stride_sn, stride_sk,
                  stride_om, stride_on,
                  BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
                  BLOCK_K: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_m = offs_m < M
    mask_n = offs_n < N
    offs_kp = tl.arange(0, BLOCK_K // 2)  # packed-byte lanes
    offs_k = tl.arange(0, BLOCK_K)
    Kp = K // 2

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    num_k = tl.cdiv(K, BLOCK_K)
    for kt in range(0, num_k):
        k0 = kt * BLOCK_K
        kp0 = kt * (BLOCK_K // 2)
        # packed int4 weights: [BLOCK_N, BLOCK_K//2] uint8, two nibbles per byte
        wp_ptrs = wp_ptr + offs_n[:, None] * stride_wn + (kp0 + offs_kp)[None, :] * stride_wk
        wp_mask = mask_n[:, None] & ((kp0 + offs_kp)[None, :] < Kp)
        b = tl.load(wp_ptrs, mask=wp_mask, other=0)
        lo = (b & 0xF).to(tl.float32) - 8.0          # even k (low nibble)
        hi = ((b >> 4) & 0xF).to(tl.float32) - 8.0   # odd  k (high nibble)
        w_nib = tl.interleave(lo, hi)                # [BLOCK_N, BLOCK_K] in k-order
        # one fp16 scale per 64-group (BLOCK_K == GROUP), broadcast over k
        s = tl.load(s_ptr + offs_n * stride_sn + kt * stride_sk,
                    mask=mask_n, other=0.0).to(tl.float32)
        w_deq = (w_nib * s[:, None]).to(tl.float16)
        # activations: [BLOCK_M, BLOCK_K]
        x_ptrs = x_ptr + offs_m[:, None] * stride_xm + (k0 + offs_k)[None, :] * stride_xk
        x_mask = mask_m[:, None] & ((k0 + offs_k)[None, :] < K)
        x = tl.load(x_ptrs, mask=x_mask, other=0.0)
        acc += tl.dot(x, tl.trans(w_deq))
    o_ptrs = o_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on
    tl.store(o_ptrs, acc.to(tl.float16), mask=mask_m[:, None] & mask_n[None, :])


def solution(x, w_packed, scales, w_fp16=None):
    x = x.contiguous()
    w_packed = w_packed.contiguous()
    scales = scales.contiguous()
    M, K = x.shape
    N, Kp = w_packed.shape
    o = torch.empty((M, N), device=x.device, dtype=torch.float16)
    BLOCK_M = 32
    BLOCK_N = 128
    BLOCK_K = GROUP  # 64 == group size, keeps one scale per k-tile
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    _w4a16_kernel[grid](
        x, w_packed, scales, o,
        M, N, K,
        x.stride(0), x.stride(1),
        w_packed.stride(0), w_packed.stride(1),
        scales.stride(0), scales.stride(1),
        o.stride(0), o.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        num_warps=4, num_stages=3,
    )
    return o

def reference(x, w_packed, scales, w_fp16=None):
    N, Kp = w_packed.shape
    lo = (w_packed & 15).float() - 8.0
    hi = (w_packed >> 4 & 15).float() - 8.0
    w = torch.stack([lo, hi], dim=-1).reshape(N, 2 * Kp)
    w = (w * scales.repeat_interleave(GROUP, dim=1).float()).half()
    return (x.float() @ w.float().T).half()

def _baseline_fp16(x, w_packed, scales, w_fp16):
    return x @ w_fp16.T

def make_inputs(preset, device, dtype):
    shapes = {'small': (32, 256, 256), 'mid': (128, 2048, 2048), 'bench': (16, 8192, 8192)}
    M, N, K = shapes[preset]
    x = torch.randn(M, K, device=device, dtype=torch.float16)
    nib = torch.randint(0, 16, (N, K), device=device, dtype=torch.uint8)
    w_packed = nib[:, 0::2] | nib[:, 1::2] << 4
    scales = (torch.rand(N, K // GROUP, device=device) * 0.02 + 0.005).half()
    w_fp16 = ((nib.float() - 8.0) * scales.repeat_interleave(GROUP, dim=1).float()).half()
    return {'x': x, 'w_packed': w_packed, 'scales': scales, 'w_fp16': w_fp16}
