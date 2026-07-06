"""Correctness checking, benchmarking, empirical roofline, and grading."""

from __future__ import annotations

import json
import statistics
from dataclasses import dataclass, asdict

import torch

from .registry import Challenge


def _dev() -> torch.device:
    if not torch.cuda.is_available():
        raise SystemExit(
            "kernel-gym needs a CUDA GPU (Triton has no macOS backend). "
            "Sync this repo to a GPU box and rerun."
        )
    return torch.device("cuda")


# ---------------------------------------------------------------- correctness

@dataclass
class CheckResult:
    challenge: str
    preset: str
    dtype: str
    passed: bool
    max_abs_err: float
    max_rel_err: float
    detail: str = ""


def _as_tensors(x):
    """Normalize a challenge output (Tensor | scalar | tuple/list) to a list of tensors."""
    if isinstance(x, (tuple, list)):
        return list(x)
    if isinstance(x, torch.Tensor):
        return [x]
    return [torch.as_tensor(x)]


def check(ch: Challenge, preset: str, dtype: torch.dtype) -> CheckResult:
    device = _dev()
    torch.manual_seed(0)
    inputs = ch.make_inputs(preset, device, dtype)
    refs = _as_tensors(ch.reference(**inputs))
    outs = _as_tensors(ch.solution(**inputs))
    if len(outs) != len(refs):
        return CheckResult(ch.id, preset, str(dtype), False, float("inf"), float("inf"),
                           f"returned {len(outs)} tensors, expected {len(refs)}")
    rtol, atol = ch.tol[dtype]
    max_abs, max_rel, ok = 0.0, 0.0, True
    for i, (out, ref) in enumerate(zip(outs, refs)):
        if out.shape != ref.shape:
            return CheckResult(ch.id, preset, str(dtype), False, float("inf"), float("inf"),
                               f"out[{i}] shape {tuple(out.shape)} != {tuple(ref.shape)}")
        ref32, out32 = ref.float(), out.float()
        abs_err = (out32 - ref32).abs()
        max_abs = max(max_abs, abs_err.max().item())
        max_rel = max(max_rel, (abs_err / ref32.abs().clamp_min(1e-6)).max().item())
        ok = ok and bool(torch.all(abs_err <= atol + rtol * ref32.abs()))
    return CheckResult(ch.id, preset, str(dtype), ok, max_abs, max_rel)


# ---------------------------------------------------------------- benchmarking

@dataclass
class BenchResult:
    challenge: str
    preset: str
    dtype: str
    triton_ms: float
    baseline_ms: float
    speedup: float
    achieved_gbps: float
    achieved_tflops: float
    pct_roof: float          # achieved vs roofline ceiling for this AI
    arithmetic_intensity: float
    grade: str


def _bench_fn(fn, **inputs) -> float:
    from triton.testing import do_bench
    return do_bench(lambda: fn(**inputs), warmup=25, rep=100)


_PEAKS: dict[str, float] = {}


def measured_peaks(device) -> tuple[float, float]:
    """Empirical (bandwidth GB/s, fp16 TFLOPS) so rooflines don't rely on spec sheets."""
    if _PEAKS:
        return _PEAKS["bw"], _PEAKS["flops"]
    from triton.testing import do_bench
    x = torch.empty(1 << 28, device=device, dtype=torch.float16)  # 512 MB
    y = torch.empty_like(x)
    ms = do_bench(lambda: y.copy_(x), warmup=10, rep=50)
    bw = 2 * x.numel() * x.element_size() / (ms * 1e-3) / 1e9
    a = torch.randn(8192, 8192, device=device, dtype=torch.float16)
    b = torch.randn(8192, 8192, device=device, dtype=torch.float16)
    ms = do_bench(lambda: a @ b, warmup=10, rep=50)
    fl = 2 * 8192**3 / (ms * 1e-3) / 1e12
    _PEAKS.update(bw=bw, flops=fl)
    return bw, fl


def bench(ch: Challenge, preset: str, dtype: torch.dtype) -> BenchResult:
    device = _dev()
    torch.manual_seed(0)
    inputs = ch.make_inputs(preset, device, dtype)

    chk = check(ch, preset, dtype)
    if not chk.passed:
        return BenchResult(ch.id, preset, str(dtype), float("nan"), float("nan"),
                           0.0, 0.0, 0.0, 0.0, 0.0, "F")

    t_ms = _bench_fn(ch.solution, **inputs)
    b_ms = _bench_fn(ch.baseline, **inputs)
    speedup = b_ms / t_ms

    fl, by = ch.flops(inputs), ch.bytes(inputs)
    gbps = by / (t_ms * 1e-3) / 1e9
    tflops = fl / (t_ms * 1e-3) / 1e12
    ai = fl / max(by, 1.0)
    peak_bw, peak_fl = measured_peaks(device)
    roof_tflops = min(peak_fl, ai * peak_bw / 1e3)  # GB/s * flop/byte -> GFLOPS -> TFLOPS
    pct = (tflops / roof_tflops * 100) if roof_tflops > 0 else 0.0

    grade = "C"
    if speedup >= ch.grade_b:
        grade = "B"
    if speedup >= ch.grade_a:
        grade = "A"
    return BenchResult(ch.id, preset, str(dtype), t_ms, b_ms, speedup,
                       gbps, tflops, pct, ai, grade)


# ---------------------------------------------------------------- report card

def report(results: list[BenchResult], path: str | None = None) -> str:
    from tabulate import tabulate
    rows = [[r.challenge, r.preset, r.dtype.replace("torch.", ""),
             f"{r.triton_ms:.3f}", f"{r.baseline_ms:.3f}", f"{r.speedup:.2f}x",
             f"{r.achieved_gbps:.0f}", f"{r.achieved_tflops:.1f}",
             f"{r.pct_roof:.0f}%", r.grade] for r in results]
    table = tabulate(rows, headers=["id", "preset", "dtype", "triton ms", "base ms",
                                    "speedup", "GB/s", "TFLOPS", "%roof", "grade"])
    grades = [r.grade for r in results]
    gpa = statistics.mean({"F": 0, "C": 2, "B": 3, "A": 4}[g] for g in grades) if grades else 0
    summary = f"\n{len(grades)} runs — A:{grades.count('A')} B:{grades.count('B')} " \
              f"C:{grades.count('C')} F:{grades.count('F')} — GPA {gpa:.2f}"
    if path:
        with open(path, "w") as f:
            json.dump([asdict(r) for r in results], f, indent=2)
    return table + summary
