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
| t4.03 | 4 | GQA FlashAttention fwd | G query heads share one KV head; compact K/V, no expansion |
| t4.04 | 4 | W4A16 dequant matmul | int4 nibbles unpacked in-register; split-K for decode shapes |
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

No GPU on hand? Run the whole thing serverless on Modal — it spins up a GPU, installs
torch (which bundles Triton on Linux), runs CPU sanity + `test` + `grade`, and tears down:

```bash
modal run modal_run.py                 # A10G by default; writes ./card.json
modal run modal_run.py --cmd test      # just correctness
```

## Results

Report cards live in `results/`. Reference solutions, all 51 correctness cases passing:

| GPU | grades | GPA | notes |
|-----|--------|-----|-------|
| A10 | 25 A / 1 B | 3.96 | fused CE 11.4×, RMSNorm 9.0×, flash bwd 8.5×, int8 GEMM at 145% of fp16 roof |
| L4 | 23 A / 2 B / 1 C* | 3.77 | same shape of wins on half the bandwidth |
| H100 | 19 A / 3 B / 3 C / 1 F* | 3.50 | bandwidth kernels hold 89–102% of roof; A10-tuned compute kernels fall off — int8 GEMM 0.61×, flash fwd 0.65× vs Hopper-tuned baselines |

\* the L4/H100 runs predate the split-K `reset_to_zero` fix (see below); the F is that bug, not the GPU.

The cross-GPU story is the roofline story: memory-bound kernels transfer, compute-bound
kernels need per-architecture tuning. Grades are per-GPU — rerun on your card.

## Cold-writing eval (blank mode)

The set doubles as an eval. `python -m gym blank --out attempts/stubs` emits
solution-stripped stubs (ground truth + spec intact, kernels removed); a model fills in
`solution()` cold; `--solutions <dir>` grades the attempt with the same harness.
Attempts can't lift a grade by cheating: reference/make_inputs tampering is diffed
against the stubs, and correctness is a hard gate with fixed tolerances.

First subject — **Claude Opus 4.8, one-shot, no GPU access, no iteration**
(`attempts/opus48/`, graded in `results/card-opus48-cold-a10.json`):

- **51/51 correctness cases passed** — including FlashAttention forward AND backward,
  cold, on the first try.
- **GPA 3.81 vs the reference solutions' 3.96** (A10).
- It beat the shipped solutions on 3 of 14 kernels: RoPE (4.7× vs 2.2× — smarter cos/sin
  indexing that skips a materialized expansion), LayerNorm backward (5.8× vs 4.9×), and
  FlashAttention backward (10.3× vs 8.5×).
- Where it lost: W4A16 (1.36×, C — fell into the same latency-bound trap the reference
  needed split-K to escape) and GQA (0.97×, B).

## War stories the harness caught

Real bugs surfaced by running on actual GPUs, kept here because they're the curriculum:

1. **bf16 GEMM cancellation floor** — unit-normal GEMM inputs make outputs O(√K); entries
   that cancel toward zero carry a √K·eps rounding floor no relative tolerance absorbs.
   Fix: scale operands by K^−¼. A CPU-only test suite can never catch this.
2. **autotune × atomics** — the autotuner re-runs each config on the same buffers, so a
   split-K kernel's `atomic_add` partials pile up and the "tuned" result is garbage.
   Fix: `reset_to_zero=["out_ptr"]` in `@autotune`. Graded F until fixed — the hard
   correctness gate working as designed.
3. **missing `.contiguous()`** — the vendored lint (KG004) flagged the shipped
   FlashAttention forward before it ever ran. Non-contiguous strides corrupt silently.

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
