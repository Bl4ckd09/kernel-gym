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

"""t3.01 — tiled GEMM with autotuning: C = A @ B.

The centerpiece compute-bound kernel. Program computes one BLOCK_M x BLOCK_N output
tile by streaming K in BLOCK_K chunks through SRAM, accumulating in fp32 registers.
Grouped/swizzled program ordering (GROUP_M) improves L2 reuse over naive row-major.

Notes lineage: "anatomy of high-performance matmul" (@cHHillee/@gordic_aleksa) — square
tiles maximize arithmetic intensity, register blocking hides latency; "75-80% of cuBLAS
takes weeks, the last 20% is where nobody operates" (@aryanvs_). Grade B at 0.55x cuBLAS
(torch.matmul), A at 0.75x — matching the notes' "achievable without Hopper intrinsics".
"""
import torch

def _configs():
    return [Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 64, 'GROUP_M': 8}, num_warps=8, num_stages=3), Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8}, num_warps=8, num_stages=3), Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8}, num_warps=8, num_stages=4), Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=4, num_stages=4), Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=4, num_stages=4), Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=4, num_stages=4)]

@autotune(configs=_configs(), key=['M', 'N', 'K'])
@jit
def _matmul_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
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

    # Wrap in-bounds so ragged edge tiles never read OOB; store is masked.
    offs_am = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)) % M
    offs_bn = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)) % N
    offs_k = tl.arange(0, BLOCK_K)
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        k_rem = K - k * BLOCK_K
        a = tl.load(a_ptrs, mask=offs_k[None, :] < k_rem, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < k_rem, other=0.0)
        acc += tl.dot(a, b, allow_tf32=False)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    c = acc.to(c_ptr.dtype.element_ty)
    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)


def solution(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    a = a.contiguous()
    b = b.contiguous()
    M, K = a.shape
    Kb, N = b.shape
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)
    grid = lambda META: (cdiv(M, META['BLOCK_M']) * cdiv(N, META['BLOCK_N']),)
    _matmul_kernel[grid](
        a, b, c,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
    )
    return c

def reference(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return a @ b

def make_inputs(preset, device, dtype):
    shapes = {'square': (512, 512, 512), 'tall': (2048, 256, 1024), 'odd': (129, 257, 193), 'bench': (4096, 4096, 4096)}
    M, N, K = shapes[preset]
    s = K ** (-0.25)
    return {'a': torch.randn(M, K, device=device, dtype=dtype) * s, 'b': torch.randn(K, N, device=device, dtype=dtype) * s}
