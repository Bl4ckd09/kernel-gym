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

from gym import Challenge, register
from gym.tri import tl, jit, autotune, Config, cdiv, require_triton


@autotune(configs=[
    Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 64, "GROUP_M": 8}, num_warps=8, num_stages=3),
    Config({"BLOCK_M": 128, "BLOCK_N": 256, "BLOCK_K": 64, "GROUP_M": 8}, num_warps=8, num_stages=3),
    Config({"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 64, "GROUP_M": 8}, num_warps=4, num_stages=4),
], key=["M", "N", "K"])
@jit
def _int8_matmul_kernel(a_ptr, b_ptr, xs_ptr, ws_ptr, c_ptr, M, N, K,
                        stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
                        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
                        GROUP_M: tl.constexpr):
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_m = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)) % M
    offs_n = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)) % N
    offs_k = tl.arange(0, BLOCK_K)
    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.int32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        kmask = offs_k[None, :] < K - k * BLOCK_K
        a = tl.load(a_ptrs, mask=kmask, other=0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_K, other=0)
        acc += tl.dot(a, b, out_dtype=tl.int32)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    xs = tl.load(xs_ptr + offs_cm, mask=offs_cm < M, other=0.0)
    ws = tl.load(ws_ptr + offs_cn, mask=offs_cn < N, other=0.0)
    out = acc.to(tl.float32) * xs[:, None] * ws[None, :]

    c_ptrs = c_ptr + offs_cm[:, None] * stride_cm + offs_cn[None, :] * stride_cn
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, out.to(c_ptr.dtype.element_ty), mask=c_mask)


def solution(xq, wq, xs, ws):
    require_triton()
    xq, wq = xq.contiguous(), wq.contiguous()
    M, K = xq.shape
    K2, N = wq.shape
    assert K == K2
    c = torch.empty((M, N), device=xq.device, dtype=torch.float16)
    grid = lambda meta: (cdiv(M, meta["BLOCK_M"]) * cdiv(N, meta["BLOCK_N"]),)
    _int8_matmul_kernel[grid](xq, wq, xs, ws, c, M, N, K,
                              xq.stride(0), xq.stride(1), wq.stride(0), wq.stride(1),
                              c.stride(0), c.stride(1))
    return c


def reference(xq, wq, xs, ws):
    acc = torch.matmul(xq.to(torch.int32).float(), wq.to(torch.int32).float())
    return (acc * xs[:, None] * ws[None, :]).to(torch.float16)


def _baseline_fp16(xq, wq, xs, ws):
    # what int8 is competing against: a plain fp16 matmul of the dequantized operands
    a = xq.float() * xs[:, None]
    b = wq.float() * ws[None, :]
    return (a.half() @ b.half())


def make_inputs(preset, device, dtype):
    shapes = {"square": (512, 512, 512), "odd": (129, 257, 193), "bench": (4096, 4096, 4096)}
    M, N, K = shapes[preset]
    xq = torch.randint(-127, 128, (M, K), device=device, dtype=torch.int8)
    wq = torch.randint(-127, 128, (K, N), device=device, dtype=torch.int8)
    xs = torch.rand(M, device=device, dtype=torch.float32) * 0.02 + 0.001
    ws = torch.rand(N, device=device, dtype=torch.float32) * 0.02 + 0.001
    return {"xq": xq, "wq": wq, "xs": xs, "ws": ws}


register(Challenge(
    id="t3.02", name="INT8 GEMM (per-row/col dequant)", tier=3,
    description="int8*int8->int32 tensor-core accumulate, fused per-row/col dequant epilogue.",
    sources=[
        "@maharshii — row-scaled int8 linear, 3.1-3.5x over bf16",
        "@prajdabre — per-row/per-column scales beat per-tensor on outlier-heavy inputs",
    ],
    make_inputs=make_inputs, reference=reference, solution=solution,
    baseline=_baseline_fp16,
    flops=lambda i: 2 * i["xq"].shape[0] * i["wq"].shape[1] * i["xq"].shape[1],
    bytes=lambda i: i["xq"].numel() + i["wq"].numel()
                    + 2 * i["xq"].shape[0] * i["wq"].shape[1],
    presets={"square": {}, "odd": {}, "bench": {}},
    dtypes=(torch.float16,),
    tol={torch.float16: (2e-3, 1e-2)},
    grade_b=1.2, grade_a=1.6,
))
