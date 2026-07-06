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

from gym import Challenge, register
from gym.tri import tl, jit, cdiv, require_triton


@jit
def _ln_dx_kernel(dy_ptr, x_ptr, w_ptr, mean_ptr, rstd_ptr, dx_ptr,
                  row_stride, n_cols, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    col = tl.arange(0, BLOCK)
    mask = col < n_cols
    dy = tl.load(dy_ptr + row * row_stride + col, mask=mask, other=0.0).to(tl.float32)
    x = tl.load(x_ptr + row * row_stride + col, mask=mask, other=0.0).to(tl.float32)
    w = tl.load(w_ptr + col, mask=mask, other=0.0).to(tl.float32)
    mean = tl.load(mean_ptr + row)
    rstd = tl.load(rstd_ptr + row)
    xhat = (x - mean) * rstd
    g = dy * w
    g = tl.where(mask, g, 0.0)
    mean_g = tl.sum(g, axis=0) / n_cols
    mean_gxhat = tl.sum(g * xhat, axis=0) / n_cols
    dx = (g - mean_g - xhat * mean_gxhat) * rstd
    tl.store(dx_ptr + row * row_stride + col, dx.to(dx_ptr.dtype.element_ty), mask=mask)


@jit
def _ln_dwdb_kernel(dy_ptr, x_ptr, mean_ptr, rstd_ptr, dw_ptr, db_ptr, M, N,
                    row_stride, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    pid = tl.program_id(0)
    cols = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    col_mask = cols < N
    dw = tl.zeros((BLOCK_N,), dtype=tl.float32)
    db = tl.zeros((BLOCK_N,), dtype=tl.float32)
    for start in range(0, M, BLOCK_M):
        rows = start + tl.arange(0, BLOCK_M)
        row_mask = rows < M
        m2 = row_mask[:, None] & col_mask[None, :]
        offs = rows[:, None] * row_stride + cols[None, :]
        dy = tl.load(dy_ptr + offs, mask=m2, other=0.0).to(tl.float32)
        x = tl.load(x_ptr + offs, mask=m2, other=0.0).to(tl.float32)
        mean = tl.load(mean_ptr + rows, mask=row_mask, other=0.0)
        rstd = tl.load(rstd_ptr + rows, mask=row_mask, other=0.0)
        xhat = (x - mean[:, None]) * rstd[:, None]
        dw += tl.sum(dy * xhat, axis=0)
        db += tl.sum(dy, axis=0)
    tl.store(dw_ptr + cols, dw, mask=col_mask)
    tl.store(db_ptr + cols, db, mask=col_mask)


def solution(dy, x, weight, mean, rstd):
    require_triton()
    dy, x = dy.contiguous(), x.contiguous()
    M, N = x.shape
    dx = torch.empty_like(x)
    dw = torch.empty(N, device=x.device, dtype=torch.float32)
    db = torch.empty(N, device=x.device, dtype=torch.float32)
    BLOCK = 1 << (N - 1).bit_length()
    nw = 16 if BLOCK >= 8192 else 8 if BLOCK >= 2048 else 4
    _ln_dx_kernel[(M,)](dy, x, weight, mean, rstd, dx, x.stride(0), N, BLOCK=BLOCK, num_warps=nw)
    BLOCK_N = 64
    _ln_dwdb_kernel[(cdiv(N, BLOCK_N),)](dy, x, mean, rstd, dw, db, M, N, x.stride(0),
                                         BLOCK_M=64, BLOCK_N=BLOCK_N)
    return dx, dw.to(weight.dtype), db.to(weight.dtype)


def reference(dy, x, weight, mean, rstd):
    xf = x.float().detach().requires_grad_(True)
    w = weight.float().detach().requires_grad_(True)
    b = torch.zeros_like(w, requires_grad=True)
    y = F.layer_norm(xf, (x.shape[-1],), w, b, eps=1e-5)
    y.backward(dy.float())
    return xf.grad.to(x.dtype), w.grad.to(weight.dtype), b.grad.to(weight.dtype)


def make_inputs(preset, device, dtype):
    shapes = {"small": (128, 256), "wide": (64, 4096), "bench": (8192, 4096)}
    M, N = shapes[preset]
    x = torch.randn(M, N, device=device, dtype=dtype)
    xf = x.float()
    mean = xf.mean(-1)
    rstd = torch.rsqrt(xf.var(-1, unbiased=False) + 1e-5)
    return {"dy": torch.randn(M, N, device=device, dtype=dtype),
            "x": x, "weight": torch.randn(N, device=device, dtype=dtype),
            "mean": mean, "rstd": rstd}


register(Challenge(
    id="t4.02", name="LayerNorm backward", tier=4,
    description="dx per-row (two reductions) + dweight/dbias column reduction; reuse stashed mean/rstd.",
    sources=[
        "@vllm_project — hand-written matching backward for train/inference parity",
        "@cloneofsimo — the backward pass is the hard, rarely-done part",
    ],
    make_inputs=make_inputs, reference=reference, solution=solution,
    flops=lambda i: 20 * i["x"].numel(),
    bytes=lambda i: 4 * i["x"].numel() * i["x"].element_size(),
    presets={"small": {}, "wide": {}, "bench": {}},
    tol={torch.float32: (2e-4, 2e-4), torch.float16: (2e-2, 2e-2),
         torch.bfloat16: (3e-2, 3e-2)},
    grade_b=0.8, grade_a=1.0,
))
