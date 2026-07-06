# kernel-gym

A graded set of **Triton kernel challenges** distilled from 505 CUDA/Triton notes in my
second brain (~305 carried real kernel/perf content). Each challenge ships a reference
PyTorch implementation, a worked Triton solution, **correctness tests**, and a
**benchmark against the PyTorch baseline** with an empirical roofline and a letter grade.

The point isn't to collect kernels — it's a **ceiling test**. The tier-5 kernels
(FlashAttention forward + backward) are exactly the "last 20% where almost nobody
operates" that the notes keep circling. Writing and reasoning about them correctly is the
bar.

## The ladder

| id | tier | kernel | the idea being tested |
|------|:----:|--------|-----------------------|
| t1.01 | 1 | fused bias + GELU | kill an HBM round-trip by fusing the pointwise epilogue |
| t1.02 | 1 | fused SwiGLU gate | `silu(a)*b` in one pass — the LLaMA gated-MLP activation |
| t2.01 | 2 | row softmax (stable) | one program per row, max-subtracted; seed of online softmax |
| t2.02 | 2 | RMSNorm forward | memory-bound row reduction, fp32 sum-of-squares |
| t2.03 | 2 | fused RoPE apply | rotate-half Q/K in one pass, cos/sin streamed |
| t3.01 | 3 | tiled GEMM (autotuned) | square SMEM tiles + register blocking + L2 swizzle |
| t3.02 | 3 | INT8 GEMM (per-row/col) | int8→int32 tensor-core accumulate, fused dequant epilogue |
| t3.03 | 3 | LayerNorm forward | mean+var in one pass; stash stats for the backward |
| t4.01 | 4 | fused cross-entropy | per-row logsumexp, never materialize the (M,V) softmax |
| t4.02 | 4 | LayerNorm backward | dx per-row + dw/db column reduction, no atomics |
| t5.01 | 5 | FlashAttention fwd (causal) | online-softmax tiling, no N×N scores in HBM |
| t5.02 | 5 | FlashAttention bwd (causal) | dq/dk/dv by recomputing softmax from stashed logsumexp |

Every challenge cites the notes it came from (`ch.sources`), e.g. FlashAttention's
Q-outer/KV-inner loop order (@pranay5255), the fused-CE memory win (@danielhanchen),
per-row/col int8 scales beating per-tensor (@prajdabre, @maharshii).

## Requirements

Triton has no macOS/CPU backend, so the kernels **run on a CUDA GPU**. On a laptop you can
still `list`, `lint`, and run the **CPU reference tests** (the shim in `gym/tri.py` makes
every module import without Triton). See [SETUP.md](SETUP.md) for the GPU box.

## Usage

```bash
python -m gym list                 # the ladder
python -m gym test  t2.01          # correctness across presets & dtypes (needs GPU)
python -m gym test                 # every challenge
python -m gym bench t5.01          # benchmark vs PyTorch baseline + roofline + grade
python -m gym grade --json card.json   # full report card
python -m gym lint  t3.01          # static kernel-quality lint (runs anywhere)

pytest tests/test_reference_cpu.py     # verify the reference logic on CPU (no GPU)
pytest -m tier5                        # correctness for one tier (needs GPU)
```

## Grading

Correctness is a hard gate (per-preset, per-dtype, with dtype-aware tolerances). Given a
correct kernel, the grade is its **speedup over the PyTorch baseline**:

- **A** — meets `grade_a` (e.g. FlashAttention fwd ≥ 0.85× of `scaled_dot_product_attention`)
- **B** — meets `grade_b`
- **C** — correct but below the bar
- **F** — incorrect

The bench also reports achieved GB/s, TFLOPS, arithmetic intensity, and **% of the
empirical roofline** — peaks are measured on *your* GPU (a copy kernel for bandwidth, a big
matmul for fp16 FLOPS), not read off a spec sheet, so "70% of roofline" means 70% of what
the silicon in front of you actually delivers.

## How it was built

Five agents read the 505 GPU/kernel notes in parallel and extracted every kernel/perf idea
with its concrete numbers and a difficulty score. The recurring, cross-corroborated ideas
became the ladder above; `NOTES.md` is the consolidated extraction with source citations.

## Layout

```
gym/
  registry.py   Challenge dataclass + auto-discovery
  harness.py    correctness check, benchmark, empirical roofline, grading
  lint.py       static kernel-quality rules (adapted from my triton-kernel-lint repo)
  tri.py        Triton import shim (import on CPU, run on GPU)
  __main__.py   the CLI
challenges/     one module per kernel, self-registering
tests/
  test_reference_cpu.py   reference-logic verification (CPU, no GPU)
  test_challenges.py      kernel correctness (GPU, parametrized by tier)
```

## Adding a challenge

Drop a module in `challenges/` that defines `make_inputs`, `reference`, `solution`, and
calls `register(Challenge(...))`. It auto-discovers into the CLI and both test suites. Keep
`reference` pure PyTorch (CPU-runnable) and add a naive cross-check to
`tests/test_reference_cpu.py`.
