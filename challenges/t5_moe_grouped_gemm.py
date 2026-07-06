"""t5.03 — MoE grouped GEMM: per-expert matmul over a sorted token batch.

The compute core of a Mixture-of-Experts FFN. Tokens have already been routed and sorted
by expert: `sorted_x` groups all of expert 0's tokens, then expert 1's, etc., and
`group_sizes[e]` says how many rows expert e owns. Each expert has its own weight matrix
W[e] of shape (K, N). Compute out[row] = sorted_x[row] @ W[expert_of_row] — one launch,
one variable-sized matmul per expert, no Python loop over experts and no padding every
expert to the same token count.

Why it matters: naive MoE runs E separate small matmuls (E kernel launches, terrible SM
utilization) or pads every expert to max tokens (wasted FLOPs). The grouped kernel maps a
flat grid of tiles onto the ragged group boundaries so one launch covers all experts with
no waste. The scheduling trick: precompute, per output m-tile, which expert it belongs to
and its local m-offset (`tile_expert`, `tile_m`), so every program is O(1) — no scan over
experts inside the kernel. The eager baseline does the Python-loop-of-matmuls this replaces.

Notes lineage: @tri_dao (SonicMoE) — fuse token gather with the grouped GEMM, 1.86x on
H100; @_xjdr (RDEP) — pool tokens to the owning rank so experts see D-times more tokens,
turning hundreds of tiny GEMMs into fewer fat ones; @UnslothAI — fused grouped-GEMM MoE
kernels, 12x; @PatrickToulme (CuTile) — grouped GEMM is the MoE building block.
"""

import torch

from gym import Challenge, register
from gym.tri import tl, jit, autotune, Config, cdiv, require_triton

BLOCK_M = 64  # fixed so the host tile schedule and the kernel agree


@autotune(configs=[
    Config({"BLOCK_N": 64, "BLOCK_K": 32}, num_warps=4, num_stages=4),
    Config({"BLOCK_N": 128, "BLOCK_K": 32}, num_warps=8, num_stages=3),
    Config({"BLOCK_N": 128, "BLOCK_K": 64}, num_warps=8, num_stages=4),
    Config({"BLOCK_N": 256, "BLOCK_K": 64}, num_warps=8, num_stages=3),
    Config({"BLOCK_N": 256, "BLOCK_K": 32}, num_warps=8, num_stages=4),
    Config({"BLOCK_N": 64, "BLOCK_K": 64}, num_warps=4, num_stages=5),
], key=["N", "K"])
@jit
def _grouped_gemm_kernel(x_ptr, w_ptr, out_ptr, starts_ptr, sizes_ptr,
                         tile_expert_ptr, tile_m_ptr, N, K,
                         stride_xm, stride_xk, stride_we, stride_wk, stride_wn,
                         stride_om, stride_on,
                         BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    pid_t = tl.program_id(0)      # which (expert, m-tile) — resolved on host
    pid_n = tl.program_id(1)      # which N-tile

    expert = tl.load(tile_expert_ptr + pid_t)
    m_tile = tl.load(tile_m_ptr + pid_t)
    start = tl.load(starts_ptr + expert)
    sz = tl.load(sizes_ptr + expert)

    offs_m = m_tile * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    row_ok = offs_m < sz
    col_ok = offs_n < N

    x_ptrs = x_ptr + (start + offs_m)[:, None] * stride_xm + offs_k[None, :] * stride_xk
    w_ptrs = w_ptr + expert * stride_we + offs_k[:, None] * stride_wk + offs_n[None, :] * stride_wn

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k0 in range(0, K, BLOCK_K):
        kk = k0 + offs_k
        x = tl.load(x_ptrs, mask=row_ok[:, None] & (kk[None, :] < K), other=0.0)
        w = tl.load(w_ptrs, mask=(kk[:, None] < K) & col_ok[None, :], other=0.0)
        acc += tl.dot(x, w)
        x_ptrs += BLOCK_K * stride_xk
        w_ptrs += BLOCK_K * stride_wk

    out_ptrs = out_ptr + (start + offs_m)[:, None] * stride_om + offs_n[None, :] * stride_on
    tl.store(out_ptrs, acc.to(out_ptr.dtype.element_ty), mask=row_ok[:, None] & col_ok[None, :])


def _tile_schedule(group_sizes: torch.Tensor):
    """Host-side: expand each expert into ceil(size/BLOCK_M) m-tiles -> flat schedule."""
    m_tiles = (group_sizes + BLOCK_M - 1) // BLOCK_M          # tiles per expert
    experts = torch.repeat_interleave(
        torch.arange(len(group_sizes), device=group_sizes.device), m_tiles)
    # local m-tile index within each expert: 0,1,..,k-1 repeated per expert
    total = int(m_tiles.sum().item())
    ends = torch.cumsum(m_tiles, 0)
    starts_t = ends - m_tiles
    tile_m = torch.arange(total, device=group_sizes.device) - starts_t.repeat_interleave(m_tiles)
    return experts.to(torch.int32), tile_m.to(torch.int32), total


def solution(sorted_x, weights, group_sizes):
    require_triton()
    sorted_x = sorted_x.contiguous()
    weights = weights.contiguous()
    M, K = sorted_x.shape
    E, K2, N = weights.shape
    starts = torch.zeros(E, device=sorted_x.device, dtype=torch.int32)
    starts[1:] = torch.cumsum(group_sizes, 0)[:-1].to(torch.int32)
    sizes = group_sizes.to(torch.int32)
    tile_expert, tile_m, num_tiles = _tile_schedule(group_sizes)
    out = torch.zeros((M, N), device=sorted_x.device, dtype=sorted_x.dtype)

    grid = lambda meta: (num_tiles, cdiv(N, meta["BLOCK_N"]))
    _grouped_gemm_kernel[grid](
        sorted_x, weights, out, starts, sizes, tile_expert, tile_m, N, K,
        sorted_x.stride(0), sorted_x.stride(1),
        weights.stride(0), weights.stride(1), weights.stride(2),
        out.stride(0), out.stride(1), BLOCK_M=BLOCK_M)
    return out


def reference(sorted_x, weights, group_sizes):
    out = torch.empty((sorted_x.shape[0], weights.shape[2]),
                      device=sorted_x.device, dtype=sorted_x.dtype)
    off = 0
    for e in range(weights.shape[0]):
        n = int(group_sizes[e].item())
        if n:
            out[off:off + n] = sorted_x[off:off + n] @ weights[e]
        off += n
    return out


def make_inputs(preset, device, dtype):
    cfg = {"small": (8, 64, 128, 128), "mid": (16, 256, 1024, 1024),
           "bench": (32, 512, 4096, 4096)}
    E, avg, K, N = cfg[preset]
    torch.manual_seed(0)
    g = torch.randint(avg // 2, avg * 3 // 2 + 1, (E,), device=device)  # uneven routing
    M = int(g.sum().item())
    return {"sorted_x": torch.randn(M, K, device=device, dtype=dtype) * 0.1,
            "weights": torch.randn(E, K, N, device=device, dtype=dtype) * 0.1,
            "group_sizes": g}


register(Challenge(
    id="t5.03", name="MoE grouped GEMM", tier=5,
    description="One launch, one variable-sized matmul per expert over a token-sorted batch; no padding.",
    sources=[
        "@tri_dao (SonicMoE) — fuse gather with grouped GEMM, 1.86x on H100",
        "@_xjdr (RDEP) — pool tokens so each expert sees D-times more; fat GEMMs beat tiny ones",
        "@UnslothAI — fused grouped-GEMM MoE kernels, 12x faster training",
    ],
    make_inputs=make_inputs, reference=reference, solution=solution,
    flops=lambda i: 2 * i["sorted_x"].shape[0] * i["weights"].shape[1] * i["weights"].shape[2],
    bytes=lambda i: (i["sorted_x"].numel() + i["weights"].numel()
                     + i["sorted_x"].shape[0] * i["weights"].shape[2]) * i["sorted_x"].element_size(),
    presets={"small": {}, "mid": {}, "bench": {}},
    dtypes=(torch.float16, torch.bfloat16),
    tol={torch.float16: (2e-2, 2e-2), torch.bfloat16: (3e-2, 3e-2)},
    # baseline is a loop of near-peak cuBLAS calls (launch overhead negligible at these
    # fat shapes); matching it with one hand-written grouped launch is the real target.
    # Grouped GEMM's structural win — one launch vs E — dominates at many-tiny-experts.
    grade_b=0.85, grade_a=1.0,
))
