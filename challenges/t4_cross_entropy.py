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

from gym import Challenge, register
from gym.tri import tl, jit, require_triton


@jit
def _cross_entropy_kernel(logits_ptr, target_ptr, loss_ptr, row_stride, V,
                          BLOCK: tl.constexpr):
    row = tl.program_id(0)
    base = logits_ptr + row * row_stride
    # pass 1: streaming max + sum-exp (online / logsumexp)
    m = -float("inf")
    s = 0.0
    for start in range(0, V, BLOCK):
        col = start + tl.arange(0, BLOCK)
        mask = col < V
        x = tl.load(base + col, mask=mask, other=-float("inf")).to(tl.float32)
        blk_max = tl.max(x, axis=0)
        new_m = tl.maximum(m, blk_max)
        s = s * tl.exp(m - new_m) + tl.sum(tl.exp(x - new_m), axis=0)
        m = new_m
    lse = m + tl.log(s)
    tgt = tl.load(target_ptr + row)
    x_tgt = tl.load(base + tgt).to(tl.float32)
    tl.store(loss_ptr + row, lse - x_tgt)


def solution(logits, target, reduction: str = "mean"):
    require_triton()
    logits = logits.contiguous()
    M, V = logits.shape
    loss = torch.empty(M, device=logits.device, dtype=torch.float32)
    BLOCK = 4096 if V >= 4096 else 1 << (V - 1).bit_length()
    _cross_entropy_kernel[(M,)](logits, target, loss, logits.stride(0), V,
                                BLOCK=BLOCK, num_warps=8)
    if reduction == "mean":
        return loss.mean()
    if reduction == "sum":
        return loss.sum()
    return loss


def reference(logits, target, reduction: str = "mean"):
    return F.cross_entropy(logits.float(), target, reduction=reduction)


def make_inputs(preset, device, dtype):
    shapes = {"small": (256, 512), "vocab": (1024, 32000), "bench": (8192, 128256)}
    M, V = shapes[preset]
    return {"logits": torch.randn(M, V, device=device, dtype=dtype),
            "target": torch.randint(0, V, (M,), device=device, dtype=torch.int64),
            "reduction": "mean"}


register(Challenge(
    id="t4.01", name="fused cross-entropy", tier=4,
    description="Per-row logsumexp minus target logit in one streaming pass; no (M,V) softmax in HBM.",
    sources=[
        "@danielhanchen — memory-efficient CE that never materializes full logits",
        "@cHHillee — logsumexp is the same online-max trick as softmax/flash",
    ],
    make_inputs=make_inputs, reference=reference, solution=solution,
    flops=lambda i: 5 * i["logits"].numel(),
    bytes=lambda i: i["logits"].numel() * i["logits"].element_size(),
    presets={"small": {}, "vocab": {}, "bench": {}},
    tol={torch.float32: (1e-4, 1e-4), torch.float16: (1e-2, 1e-2),
         torch.bfloat16: (2e-2, 2e-2)},
    grade_b=0.9, grade_a=1.1,
))
