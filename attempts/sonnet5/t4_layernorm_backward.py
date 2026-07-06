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


@jit
def _ln_bwd_dx_kernel(
    dy_ptr, x_ptr, w_ptr, mean_ptr, rstd_ptr, dx_ptr,
    N,
    stride_m,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N

    dy_row = dy_ptr + row * stride_m
    x_row = x_ptr + row * stride_m
    dx_row = dx_ptr + row * stride_m

    x = tl.load(x_row + cols, mask=mask, other=0.0).to(tl.float32)
    dyv = tl.load(dy_row + cols, mask=mask, other=0.0).to(tl.float32)
    w = tl.load(w_ptr + cols, mask=mask, other=0.0).to(tl.float32)

    mean_m = tl.load(mean_ptr + row).to(tl.float32)
    rstd_m = tl.load(rstd_ptr + row).to(tl.float32)

    xhat = (x - mean_m) * rstd_m
    g = dyv * w

    n_f = N.to(tl.float32)
    sum_g = tl.sum(g, axis=0)
    sum_gxhat = tl.sum(g * xhat, axis=0)
    mean_g = sum_g / n_f
    mean_gxhat = sum_gxhat / n_f

    dx = rstd_m * (g - mean_g - xhat * mean_gxhat)

    tl.store(dx_row + cols, dx.to(dx_ptr.dtype.element_ty), mask=mask)


@jit
def _ln_bwd_dwdb_kernel(
    dy_ptr, x_ptr, mean_ptr, rstd_ptr, dw_ptr, db_ptr,
    M, N,
    stride_m,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    pid = tl.program_id(0)
    cols = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    col_mask = cols < N

    acc_dw = tl.zeros((BLOCK_N,), dtype=tl.float32)
    acc_db = tl.zeros((BLOCK_N,), dtype=tl.float32)

    for row_start in range(0, M, BLOCK_M):
        rows = row_start + tl.arange(0, BLOCK_M)
        row_mask = rows < M
        mask2d = row_mask[:, None] & col_mask[None, :]

        ptrs_x = x_ptr + rows[:, None] * stride_m + cols[None, :]
        ptrs_dy = dy_ptr + rows[:, None] * stride_m + cols[None, :]

        x = tl.load(ptrs_x, mask=mask2d, other=0.0).to(tl.float32)
        dyv = tl.load(ptrs_dy, mask=mask2d, other=0.0).to(tl.float32)

        mean_r = tl.load(mean_ptr + rows, mask=row_mask, other=0.0).to(tl.float32)
        rstd_r = tl.load(rstd_ptr + rows, mask=row_mask, other=0.0).to(tl.float32)

        xhat = (x - mean_r[:, None]) * rstd_r[:, None]

        acc_dw += tl.sum(dyv * xhat, axis=0)
        acc_db += tl.sum(dyv, axis=0)

    tl.store(dw_ptr + cols, acc_dw, mask=col_mask)
    tl.store(db_ptr + cols, acc_db, mask=col_mask)


def solution(dy, x, weight, mean, rstd):
    """YOUR KERNEL HERE — see rules at top of file."""
    dy = dy.contiguous()
    x = x.contiguous()
    weight = weight.contiguous()
    mean = mean.contiguous()
    rstd = rstd.contiguous()

    M, N = x.shape
    dx = torch.empty_like(x)
    dw = torch.empty(N, device=x.device, dtype=torch.float32)
    db = torch.empty(N, device=x.device, dtype=torch.float32)

    BLOCK_N_DX = triton.next_power_of_2(N)
    num_warps = 4
    if BLOCK_N_DX >= 2048:
        num_warps = 8
    if BLOCK_N_DX >= 8192:
        num_warps = 16

    _ln_bwd_dx_kernel[(M,)](
        dy, x, weight, mean, rstd, dx,
        N,
        x.stride(0),
        BLOCK_N=BLOCK_N_DX,
        num_warps=num_warps,
    )

    BLOCK_M = 32
    BLOCK_N = 64
    grid = (cdiv(N, BLOCK_N),)
    _ln_bwd_dwdb_kernel[grid](
        dy, x, mean, rstd, dw, db,
        M, N,
        x.stride(0),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
    )

    return dx, dw.to(weight.dtype), db.to(weight.dtype)

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
