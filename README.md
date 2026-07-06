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
| t4.05 | 4 | sliding-window attention | banded causal; out-of-window K/V blocks never loaded (O(N·W)) |
| t5.01 | 5 | FlashAttention fwd (causal) | online-softmax tiling, no N×N scores in HBM |
| t5.02 | 5 | FlashAttention bwd (causal) | dq/dk/dv by recomputing softmax from stashed logsumexp |
| t5.03 | 5 | MoE grouped GEMM | per-expert matmul over a token-sorted batch, one launch, no padding |

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

Each model gets the 16 stubs cold — one shot, no GPU access, no iteration, no sight of the
reference solutions — and four agents (by tier) fill them in. Graded on A10:

| writer | correctness | GPA | A / B / C / F |
|--------|-------------|-----|---------------|
| **reference** (hand-written) | 30/30 | **3.87** | 27 / 2 / 1 / 0 |
| **Claude Sonnet 5** (cold) | 30/30 | **3.73** | 25 / 2 / 3 / 0 |
| **Claude Opus 4.8** (cold) | 30/30 | **3.70** | 24 / 3 / 3 / 0 |

The headline: **both models wrote all 16 kernels correct on the first try — zero failures**,
including FlashAttention forward *and* backward, GQA, sliding-window (with its NaN trap),
and MoE grouped GEMM. Cold, blind, no GPU to test against. And they land within ~0.15 GPA
of hand-tuned reference solutions.

Where the grades diverge (cases where the three differ, speedup in ×):

| case | reference | Opus 4.8 | Sonnet 5 |
|------|-----------|----------|----------|
| GQA fwd | A (1.03) | B (1.00) | B (0.92) |
| W4A16 | B (1.52) | C (1.36) | C (1.30) |
| flash fwd bf16 | A (0.96) | B (0.84) | A (0.92) |
| MoE grouped GEMM | B (0.92) | C (0.83) | C (0.39) |

The gap is entirely in the two hardest quantitative kernels — W4A16 (both models missed
the split-K trick that saves the decode shape) and MoE grouped GEMM (matching a loop of
cuBLAS calls is the frontier; Sonnet's schedule was the weakest here). On the bandwidth
kernels and even the flash-attention pair, cold model output is at or near hand-tuned. In
earlier runs Opus *beat* the reference on RoPE, LayerNorm backward, and FlashAttention
backward — the models aren't just reproducing textbook kernels, they're finding wins.

## KernelBench interop

kernel-gym challenges can be exported into [KernelBench](https://github.com/ScalingIntelligence/KernelBench)'s
task format so an external, standard harness can grade them:

```bash
python -m gym export --out kernelbench_export     # one task file per challenge + manifest.json
python -m gym export --out DIR --preset bench      # emit a specific input preset
```

Each challenge becomes a self-contained KernelBench task under `DIR/level{N}/{id}.py`:

- `class Model(nn.Module)` — `forward` wraps the challenge's PyTorch `reference()`,
- `class ModelNew(nn.Module)` — `forward` wraps the Triton `solution()`,
- `get_inputs()` — calls the challenge's `make_inputs(preset, 'cuda', dtype)` and returns the
  tensor args in forward order; `get_init_inputs()` returns `[]`.

Non-tensor challenge args (`eps`, `causal`, `window`, `reduction`, `sm_scale`, …) are baked
into the module as constructor state so `forward` takes only tensors — which are vs. aren't
tensors is decided by `torch.is_tensor` at export time. `manifest.json` lists every task as
`{id, name, level, tier, preset, dtype, file}`. The emitted files import triton and only
*run* on a CUDA box (export itself needs no GPU).

**Level mapping** (kernel-gym tier → KernelBench level): tiers 1–2 → **level 1**,
tier 3 → **level 2**, tiers 4–5 → **level 3**. Current set exports to **5 / 3 / 8** tasks
across levels 1 / 2 / 3.

**How the two harnesses relate.** They measure complementary things. kernel-gym grades
*speedup vs. the PyTorch baseline* (`reference` or a stated `baseline`, e.g. cuBLAS matmul,
SDPA flash attention, fused CE) **plus an empirical roofline** (achieved GB/s, TFLOPS,
arithmetic intensity, % of the peaks measured on your GPU) on a small hand-picked ladder.
KernelBench grades *speedup vs. eager PyTorch* over its 250-task suite (the target named in
`NOTES.md`, where CUDA-L1 reported 17.7× avg / 449× max). Exporting lets the same kernels be
scored by KernelBench's external standard while kernel-gym keeps the roofline / per-GPU story.

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
