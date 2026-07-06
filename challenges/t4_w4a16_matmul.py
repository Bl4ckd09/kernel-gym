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

from gym import Challenge, register
from gym.tri import tl, jit, cdiv, require_triton

GROUP = 64  # K-elements per scale; == BLOCK_K so each K-step uses exactly one scale/col


from gym.tri import autotune, Config


@autotune(configs=[
    Config({"BLOCK_N": 32}, num_warps=2, num_stages=5),
    Config({"BLOCK_N": 64}, num_warps=4, num_stages=4),
    Config({"BLOCK_N": 128}, num_warps=4, num_stages=4),
], key=["M", "N", "K", "SPLIT_K"],
    # autotune re-runs each config on the same buffer; without zeroing between
    # runs the atomic_add partials pile up and the tuned result is garbage
    reset_to_zero=["out_ptr"])
@jit
def _w4a16_kernel(x_ptr, wp_ptr, s_ptr, out_ptr, M, N, K,
                  stride_xm, stride_xk, stride_wn, stride_wk,
                  stride_sn, stride_sk, stride_om, stride_on,
                  BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
                  SPLIT_K: tl.constexpr):
    # split-K: decode shapes (M~16) launch too few programs to hide latency; each of
    # SPLIT_K programs reduces a K-slice and atomically adds its partial into fp32 out.
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_s = tl.program_id(2)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    offs_kp = tl.arange(0, BLOCK_K // 2)      # packed-K offsets (2 values per byte)

    kb_per_split = tl.cdiv(tl.cdiv(K, BLOCK_K), SPLIT_K)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for i in range(0, kb_per_split):
        kb = pid_s * kb_per_split + i
        k0 = kb * BLOCK_K
        x_mask = (offs_m[:, None] < M) & (k0 + offs_k[None, :] < K)
        x = tl.load(x_ptr + offs_m[:, None] * stride_xm + (k0 + offs_k[None, :]) * stride_xk,
                    mask=x_mask, other=0.0)
        wp_mask = (offs_n[:, None] < N) & (k0 // 2 + offs_kp[None, :] < K // 2)
        packed = tl.load(wp_ptr + offs_n[:, None] * stride_wn
                         + (k0 // 2 + offs_kp[None, :]) * stride_wk,
                         mask=wp_mask, other=0)
        lo = (packed & 0xF).to(tl.float16)            # k = 2b
        hi = ((packed >> 4) & 0xF).to(tl.float16)     # k = 2b + 1
        w = tl.interleave(lo, hi)                     # (BLOCK_N, BLOCK_K) in K order
        s = tl.load(s_ptr + offs_n * stride_sn + kb * stride_sk,
                    mask=(offs_n < N) & (kb * BLOCK_K < K), other=0.0).to(tl.float16)
        w = (w - 8.0) * s[:, None]
        acc += tl.dot(x, tl.trans(w))
    out_ptrs = out_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on
    out_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    if SPLIT_K == 1:
        tl.store(out_ptrs, acc, mask=out_mask)
    else:
        tl.atomic_add(out_ptrs, acc, mask=out_mask)


def solution(x, w_packed, scales, w_fp16=None):
    require_triton()
    x, w_packed, scales = x.contiguous(), w_packed.contiguous(), scales.contiguous()
    M, K = x.shape
    N = w_packed.shape[0]
    BLOCK_M = 16 if M <= 16 else 32
    # enough K-splits that (N-tiles * splits) comfortably oversubscribes the SMs
    SPLIT_K = max(1, min(8, K // 1024)) if M <= 32 else 1
    buf = torch.zeros((M, N), device=x.device, dtype=torch.float32)
    grid = lambda meta: (cdiv(M, BLOCK_M), cdiv(N, meta["BLOCK_N"]), SPLIT_K)
    _w4a16_kernel[grid](x, w_packed, scales, buf, M, N, K,
                        x.stride(0), x.stride(1), w_packed.stride(0), w_packed.stride(1),
                        scales.stride(0), scales.stride(1), buf.stride(0), buf.stride(1),
                        BLOCK_M=BLOCK_M, BLOCK_K=GROUP, SPLIT_K=SPLIT_K)
    return buf.to(torch.float16)


def reference(x, w_packed, scales, w_fp16=None):
    # dequantize exactly as the kernel must: (nibble - 8) * per-group scale, in fp16
    N, Kp = w_packed.shape
    lo = (w_packed & 0xF).float() - 8.0
    hi = ((w_packed >> 4) & 0xF).float() - 8.0
    w = torch.stack([lo, hi], dim=-1).reshape(N, 2 * Kp)          # K order restored
    w = (w * scales.repeat_interleave(GROUP, dim=1).float()).half()
    return (x.float() @ w.float().T).half()


def _baseline_fp16(x, w_packed, scales, w_fp16):
    return x @ w_fp16.T


def make_inputs(preset, device, dtype):
    shapes = {"small": (32, 256, 256), "mid": (128, 2048, 2048), "bench": (16, 8192, 8192)}
    M, N, K = shapes[preset]
    x = torch.randn(M, K, device=device, dtype=torch.float16)
    nib = torch.randint(0, 16, (N, K), device=device, dtype=torch.uint8)
    w_packed = nib[:, 0::2] | (nib[:, 1::2] << 4)                 # (N, K//2)
    scales = (torch.rand(N, K // GROUP, device=device) * 0.02 + 0.005).half()
    w_fp16 = ((nib.float() - 8.0)
              * scales.repeat_interleave(GROUP, dim=1).float()).half()
    return {"x": x, "w_packed": w_packed, "scales": scales, "w_fp16": w_fp16}


register(Challenge(
    id="t4.04", name="W4A16 dequant matmul", tier=4,
    description="int4 weights unpacked in-register (nibbles -> interleave -> dequant -> dot); fp16 activations.",
    sources=[
        "@jeremyphoward/@Yuchenj_UW — Kimi K2 native INT4 QAT on Marlin int4 kernels",
        "@doodlestein — int4 halves decode latency; bandwidth-bound, advantage grows with context",
        "@maharshii — pack 2xint4 per uint8; unpack with & 0xF and >> 4",
    ],
    make_inputs=make_inputs, reference=reference, solution=solution,
    baseline=_baseline_fp16,
    flops=lambda i: 2 * i["x"].shape[0] * i["w_packed"].shape[0] * i["x"].shape[1],
    bytes=lambda i: (i["w_packed"].numel() + 2 * i["scales"].numel()
                     + 2 * i["x"].numel()
                     + 2 * i["x"].shape[0] * i["w_packed"].shape[0]),
    presets={"small": {}, "mid": {}, "bench": {}},
    dtypes=(torch.float16,),
    tol={torch.float16: (2e-2, 2e-2)},
    grade_b=1.5, grade_a=2.2,
))
