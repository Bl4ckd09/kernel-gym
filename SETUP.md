# Running kernel-gym on a GPU

Triton needs a CUDA GPU (no macOS/CPU backend). Any recent NVIDIA GPU works; the tier-3+
kernels use tensor cores so Ampere (A100/RTX 30xx) or newer is ideal.

## Install

```bash
git clone <this repo> && cd kernel-gym
python -m venv .venv && source .venv/bin/activate     # Python 3.10–3.12
pip install torch triton pytest tabulate              # a CUDA torch build
```

`triton` ships with the CUDA `torch` wheels; if `import triton` fails, `pip install triton`
explicitly. Verify:

```bash
python -c "import torch, triton; print(torch.cuda.get_device_name(), triton.__version__)"
```

## Sanity check, then run

```bash
pytest tests/test_reference_cpu.py    # reference logic (passes even before the GPU)
python -m gym test                    # kernel correctness on GPU (all tiers)
python -m gym grade --json card.json  # benchmarks + grades vs PyTorch
```

Correctness first, then grades. A kernel that fails `test` is graded **F** regardless of
speed.

## Notes on the harness

- **Empirical roofline.** On first `bench`, the harness measures your GPU's real copy
  bandwidth and fp16 matmul throughput, and reports achieved perf as a fraction of *that*.
  Numbers are per-GPU; don't compare grades across different cards.
- **Autotuned kernels** (t3.01, t3.02) sweep configs on first launch — the first bench of a
  new shape is slow; `do_bench` warms up before timing.
- **Grade thresholds** live on each `Challenge` (`grade_a`/`grade_b`) and are calibrated to
  the note's own claim (e.g. int8 GEMM targets ≥1.2× fp16 per @maharshii's 3.1–3.5× on a
  4060; scale to your GPU). Tune them in the challenge module if your baseline differs.

## From a Mac

This repo was authored on a GPU-less Mac. The workflow: edit + `lint` + `pytest
tests/test_reference_cpu.py` locally, then `rsync`/push to the GPU box and run `gym test` /
`gym grade` there. The Triton shim (`gym/tri.py`) is what lets the modules import on the
Mac at all.
