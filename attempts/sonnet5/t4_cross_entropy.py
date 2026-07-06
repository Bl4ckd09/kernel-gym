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


@autotune(
    configs=[
        Config({'BLOCK_V': 1024}, num_warps=4),
        Config({'BLOCK_V': 2048}, num_warps=8),
        Config({'BLOCK_V': 4096}, num_warps=8),
        Config({'BLOCK_V': 8192}, num_warps=16),
    ],
    key=['V'],
)
@jit
def _cross_entropy_fwd_kernel(
    logits_ptr, target_ptr, loss_ptr,
    V,
    stride_m,
    BLOCK_V: tl.constexpr,
):
    row = tl.program_id(0)
    row_ptr = logits_ptr + row * stride_m

    m_i = float('-inf')
    l_i = 0.0

    for start in range(0, V, BLOCK_V):
        cols = start + tl.arange(0, BLOCK_V)
        mask = cols < V
        x = tl.load(row_ptr + cols, mask=mask, other=float('-inf')).to(tl.float32)
        block_max = tl.max(x, axis=0)
        m_new = tl.maximum(m_i, block_max)
        alpha = tl.exp(m_i - m_new)
        l_i = l_i * alpha + tl.sum(tl.exp(x - m_new), axis=0)
        m_i = m_new

    logsumexp = m_i + tl.log(l_i)

    tgt = tl.load(target_ptr + row)
    tgt_logit = tl.load(row_ptr + tgt).to(tl.float32)

    loss = logsumexp - tgt_logit
    tl.store(loss_ptr + row, loss)


def solution(logits, target, reduction: str='mean'):
    """YOUR KERNEL HERE — see rules at top of file."""
    logits = logits.contiguous()
    target = target.contiguous()
    M, V = logits.shape

    loss = torch.empty(M, device=logits.device, dtype=torch.float32)

    grid = (M,)
    _cross_entropy_fwd_kernel[grid](
        logits, target, loss,
        V,
        logits.stride(0),
    )

    if reduction == 'mean':
        return loss.mean()
    elif reduction == 'sum':
        return loss.sum()
    elif reduction == 'none':
        return loss
    else:
        raise ValueError(f'unknown reduction: {reduction}')

def reference(logits, target, reduction: str='mean'):
    return F.cross_entropy(logits.float(), target, reduction=reduction)

def make_inputs(preset, device, dtype):
    shapes = {'small': (256, 512), 'vocab': (1024, 32000), 'bench': (8192, 128256)}
    M, V = shapes[preset]
    return {'logits': torch.randn(M, V, device=device, dtype=dtype), 'target': torch.randint(0, V, (M,), device=device, dtype=torch.int64), 'reduction': 'mean'}
