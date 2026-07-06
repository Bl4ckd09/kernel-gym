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

"""t3.02 — INT8 GEMM with per-row / per-column dequant.

Weights are pre-quantized per output column (wq int8 + ws fp32), activations per row
(xq int8 + xs fp32). The kernel accumulates int8*int8 -> int32 in tensor cores, then
dequantizes in the epilogue: out[m,n] = acc[m,n] * xs[m] * ws[n]. This is the shape
that wins on real hardware because int8 tensor-core throughput is ~2x fp16 and the
weights are half the bytes.

Notes lineage: @maharshii — "row-scaled int8 linear, 3.1-3.5x over bf16 on RTX 4060";
@prajdabre — "the activation-outlier trap: per-tensor int8 destroys accuracy, per-row /
per-column scales recover it". Correctness here is exact vs a PyTorch int-matmul of the
SAME quantized inputs (the challenge is the kernel, not the quantizer), so tolerance is
tight — any mismatch means the accumulate or dequant is wrong.
"""
import torch

def _int8_configs():
    return [
        Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 64, 'GROUP_M': 8}, num_warps=8, num_stages=3),
        Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8}, num_warps=8, num_stages=3),
        Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 128, 'GROUP_M': 8}, num_warps=8, num_stages=3),
        Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8}, num_warps=8, num_stages=4),
        Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8}, num_warps=4, num_stages=4),
        Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 64, 'GROUP_M': 8}, num_warps=4, num_stages=4),
    ]


@autotune(configs=_int8_configs(), key=['M', 'N', 'K'])
@jit
def _int8_matmul_kernel(
    xq_ptr, wq_ptr, xs_ptr, ws_ptr, c_ptr,
    M, N, K,
    stride_xm, stride_xk,
    stride_wk, stride_wn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_am = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)) % M
    offs_bn = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)) % N
    offs_k = tl.arange(0, BLOCK_K)
    x_ptrs = xq_ptr + (offs_am[:, None] * stride_xm + offs_k[None, :] * stride_xk)
    w_ptrs = wq_ptr + (offs_k[:, None] * stride_wk + offs_bn[None, :] * stride_wn)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.int32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        k_rem = K - k * BLOCK_K
        a = tl.load(x_ptrs, mask=offs_k[None, :] < k_rem, other=0)
        b = tl.load(w_ptrs, mask=offs_k[:, None] < k_rem, other=0)
        acc += tl.dot(a, b, out_dtype=tl.int32)
        x_ptrs += BLOCK_K * stride_xk
        w_ptrs += BLOCK_K * stride_wk

    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    xs = tl.load(xs_ptr + offs_cm, mask=offs_cm < M, other=0.0).to(tl.float32)
    ws = tl.load(ws_ptr + offs_cn, mask=offs_cn < N, other=0.0).to(tl.float32)
    out = acc.to(tl.float32) * xs[:, None] * ws[None, :]
    out = out.to(tl.float16)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, out, mask=c_mask)


def solution(xq, wq, xs, ws):
    xq = xq.contiguous()
    wq = wq.contiguous()
    xs = xs.contiguous()
    ws = ws.contiguous()
    M, K = xq.shape
    Kb, N = wq.shape
    c = torch.empty((M, N), device=xq.device, dtype=torch.float16)
    grid = lambda META: (cdiv(M, META['BLOCK_M']) * cdiv(N, META['BLOCK_N']),)
    _int8_matmul_kernel[grid](
        xq, wq, xs, ws, c,
        M, N, K,
        xq.stride(0), xq.stride(1),
        wq.stride(0), wq.stride(1),
        c.stride(0), c.stride(1),
    )
    return c

def reference(xq, wq, xs, ws):
    acc = torch.matmul(xq.to(torch.int32).float(), wq.to(torch.int32).float())
    return (acc * xs[:, None] * ws[None, :]).to(torch.float16)

def _baseline_fp16(xq, wq, xs, ws):
    a = xq.float() * xs[:, None]
    b = wq.float() * ws[None, :]
    return a.half() @ b.half()

def make_inputs(preset, device, dtype):
    shapes = {'square': (512, 512, 512), 'odd': (129, 257, 193), 'bench': (4096, 4096, 4096)}
    M, N, K = shapes[preset]
    xq = torch.randint(-127, 128, (M, K), device=device, dtype=torch.int8)
    wq = torch.randint(-127, 128, (K, N), device=device, dtype=torch.int8)
    xs = torch.rand(M, device=device, dtype=torch.float32) * 0.02 + 0.001
    ws = torch.rand(N, device=device, dtype=torch.float32) * 0.02 + 0.001
    return {'xq': xq, 'wq': wq, 'xs': xs, 'ws': ws}
