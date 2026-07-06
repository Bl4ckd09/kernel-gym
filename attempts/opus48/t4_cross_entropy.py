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

"""t4.01 — fused cross-entropy over logits (no materialized softmax).

Given logits (M, V) and integer targets (M,), compute per-row loss
  loss[m] = logsumexp(logits[m]) - logits[m, target[m]]
in a single streaming pass per row. The point is memory: the eager path materializes an
(M, V) softmax/log_softmax tensor — for a vocab of 128k that dwarfs the logits themselves.
The fused kernel keeps only running (max, sum) scalars in registers, so it reads logits
once and writes M floats. This is the single biggest activation-memory win in LLM training.

Notes lineage: @danielhanchen (Unsloth) — "memory-efficient fused cross-entropy that never
materializes the full logits"; the same online-max/logsumexp trick as softmax and flash
attention. Grade = speedup over eager F.cross_entropy (which is already fused in PyTorch,
so beating it means a genuinely tight kernel).
"""
import torch
import torch.nn.functional as F

@triton.jit
def _cross_entropy_kernel(logits_ptr, target_ptr, loss_ptr,
                          stride_m, stride_v, V,
                          BLOCK_V: tl.constexpr):
    row = tl.program_id(0)
    row_ptr = logits_ptr + row.to(tl.int64) * stride_m
    # online logsumexp: running (max, sum) over the vocab dim
    m = -float('inf')
    s = 0.0
    for start in range(0, V, BLOCK_V):
        offs = start + tl.arange(0, BLOCK_V)
        mask = offs < V
        x = tl.load(row_ptr + offs * stride_v, mask=mask,
                    other=-float('inf')).to(tl.float32)
        blk_max = tl.max(x, axis=0)
        new_m = tl.maximum(m, blk_max)
        s = s * tl.exp(m - new_m) + tl.sum(tl.exp(x - new_m), axis=0)
        m = new_m
    lse = m + tl.log(s)
    tgt = tl.load(target_ptr + row).to(tl.int64)
    x_tgt = tl.load(row_ptr + tgt * stride_v).to(tl.float32)
    tl.store(loss_ptr + row, lse - x_tgt)


def solution(logits, target, reduction: str='mean'):
    assert logits.dim() == 2
    M, V = logits.shape
    target = target.contiguous()
    loss = torch.empty((M,), device=logits.device, dtype=torch.float32)
    BLOCK_V = 4096
    _cross_entropy_kernel[(M,)](
        logits, target, loss,
        logits.stride(0), logits.stride(1), V,
        BLOCK_V=BLOCK_V, num_warps=8,
    )
    if reduction == 'mean':
        return loss.mean()
    elif reduction == 'sum':
        return loss.sum()
    else:
        return loss

def reference(logits, target, reduction: str='mean'):
    return F.cross_entropy(logits.float(), target, reduction=reduction)

def make_inputs(preset, device, dtype):
    shapes = {'small': (256, 512), 'vocab': (1024, 32000), 'bench': (8192, 128256)}
    M, V = shapes[preset]
    return {'logits': torch.randn(M, V, device=device, dtype=dtype), 'target': torch.randint(0, V, (M,), device=device, dtype=torch.int64), 'reduction': 'mean'}
