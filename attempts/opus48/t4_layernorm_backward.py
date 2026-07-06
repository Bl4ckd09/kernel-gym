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

"""t4.02 — LayerNorm backward (dx, dweight, dbias).

The gradient nobody wants to write by hand. Given dy, x, weight, and the stashed
(mean, rstd) from the forward (t3.03), produce all three grads. Two distinct patterns:

  dx  — a per-row computation. With xhat = (x-mean)*rstd and g = dy*weight,
        dx = rstd * (g - mean(g) - xhat * mean(g*xhat)). Two reductions per row over
        the feature dim, done on the SRAM-resident row. This is the memory-bound part.

  dw, db — reductions DOWN the rows (dw[n] = sum_m dy[m,n]*xhat[m,n], db[n] = sum_m dy[m,n]).
        A second kernel tiles the M dimension so each program owns a column block and
        streams all rows, accumulating in fp32. No atomics, deterministic reduction order.

Notes lineage: @vllm_project — "wrote matching custom backward passes by hand for
train/inference parity"; @cloneofsimo — "the backward pass is the hard, rarely-done part";
@aryagxr LayerNorm optimization. Grade = speedup over autograd's backward.
"""
import torch
import torch.nn.functional as F

@triton.jit
def _ln_bwd_dx_kernel(dy_ptr, x_ptr, w_ptr, mean_ptr, rstd_ptr, dx_ptr,
                      stride_m, stride_n, N,
                      BLOCK_N: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N
    off = row.to(tl.int64) * stride_m + cols * stride_n
    x = tl.load(x_ptr + off, mask=mask, other=0.0).to(tl.float32)
    dy = tl.load(dy_ptr + off, mask=mask, other=0.0).to(tl.float32)
    w = tl.load(w_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    mean = tl.load(mean_ptr + row).to(tl.float32)
    rstd = tl.load(rstd_ptr + row).to(tl.float32)
    xhat = (x - mean) * rstd
    g = dy * w  # masked lanes -> g == 0, so they drop out of the reductions
    mean_g = tl.sum(g, axis=0) / N
    mean_gxhat = tl.sum(g * xhat, axis=0) / N
    dx = rstd * (g - mean_g - xhat * mean_gxhat)
    tl.store(dx_ptr + off, dx, mask=mask)


@triton.jit
def _ln_bwd_dwdb_kernel(dy_ptr, x_ptr, mean_ptr, rstd_ptr, dw_ptr, db_ptr,
                        M, N, stride_m, stride_n,
                        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    pid = tl.program_id(0)
    cols = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_n = cols < N
    dw = tl.zeros((BLOCK_N,), dtype=tl.float32)
    db = tl.zeros((BLOCK_N,), dtype=tl.float32)
    for start in range(0, M, BLOCK_M):
        rows = start + tl.arange(0, BLOCK_M)
        mask_m = rows < M
        off = rows[:, None].to(tl.int64) * stride_m + cols[None, :] * stride_n
        m2 = mask_m[:, None] & mask_n[None, :]
        dy = tl.load(dy_ptr + off, mask=m2, other=0.0).to(tl.float32)
        x = tl.load(x_ptr + off, mask=m2, other=0.0).to(tl.float32)
        mean = tl.load(mean_ptr + rows, mask=mask_m, other=0.0).to(tl.float32)
        rstd = tl.load(rstd_ptr + rows, mask=mask_m, other=0.0).to(tl.float32)
        xhat = (x - mean[:, None]) * rstd[:, None]
        dw += tl.sum(dy * xhat, axis=0)
        db += tl.sum(dy, axis=0)
    tl.store(dw_ptr + cols, dw, mask=mask_n)
    tl.store(db_ptr + cols, db, mask=mask_n)


def solution(dy, x, weight, mean, rstd):
    x = x.contiguous()
    dy = dy.contiguous()
    weight = weight.contiguous()
    mean = mean.contiguous()
    rstd = rstd.contiguous()
    M, N = x.shape
    dx = torch.empty_like(x)
    dw = torch.empty((N,), device=x.device, dtype=weight.dtype)
    db = torch.empty((N,), device=x.device, dtype=weight.dtype)

    BLOCK_N_dx = triton.next_power_of_2(N)
    num_warps = 4 if BLOCK_N_dx <= 1024 else 8
    _ln_bwd_dx_kernel[(M,)](
        dy, x, weight, mean, rstd, dx,
        x.stride(0), x.stride(1), N,
        BLOCK_N=BLOCK_N_dx, num_warps=num_warps,
    )

    BLOCK_N = 64
    BLOCK_M = 128
    grid = (triton.cdiv(N, BLOCK_N),)
    _ln_bwd_dwdb_kernel[grid](
        dy, x, mean, rstd, dw, db,
        M, N, x.stride(0), x.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
    )
    return dx, dw, db

def reference(dy, x, weight, mean, rstd):
    xf = x.float().detach().requires_grad_(True)
    w = weight.float().detach().requires_grad_(True)
    b = torch.zeros_like(w, requires_grad=True)
    y = F.layer_norm(xf, (x.shape[-1],), w, b, eps=1e-05)
    y.backward(dy.float())
    return (xf.grad.to(x.dtype), w.grad.to(weight.dtype), b.grad.to(weight.dtype))

def make_inputs(preset, device, dtype):
    shapes = {'small': (128, 256), 'wide': (64, 4096), 'bench': (8192, 4096)}
    M, N = shapes[preset]
    x = torch.randn(M, N, device=device, dtype=dtype)
    xf = x.float()
    mean = xf.mean(-1)
    rstd = torch.rsqrt(xf.var(-1, unbiased=False) + 1e-05)
    return {'dy': torch.randn(M, N, device=device, dtype=dtype), 'x': x, 'weight': torch.randn(N, device=device, dtype=dtype), 'mean': mean, 'rstd': rstd}
