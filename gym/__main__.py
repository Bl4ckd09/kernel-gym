"""CLI: python -m gym {list|test|bench|grade} [challenge-id] [--json out.json]"""

from __future__ import annotations

import argparse
import sys

import torch

from . import registry
from . import harness


def main() -> int:
    p = argparse.ArgumentParser(prog="gym")
    p.add_argument("cmd", choices=["list", "test", "bench", "grade", "lint", "blank"])
    p.add_argument("id", nargs="?", help="challenge id, e.g. t3.02 (default: all)")
    p.add_argument("--preset", default=None, help="input preset (default: all for test, bench for bench)")
    p.add_argument("--dtype", default=None, choices=["fp32", "fp16", "bf16"])
    p.add_argument("--json", default=None, help="write bench results to JSON")
    p.add_argument("--out", default="attempts/stubs", help="output dir for blank stubs")
    p.add_argument("--solutions", default=None,
                   help="dir of attempt modules; their solution() replaces the shipped one")
    args = p.parse_args()

    chs = registry.load_all()

    if args.cmd == "blank":
        from . import blank as blankmod
        for path in blankmod.emit_stubs(args.out):
            print(f"  wrote {path}")
        return 0

    if args.solutions:
        from . import blank as blankmod
        blankmod.load_solutions(chs, args.solutions)
        print(f"  [eval mode] solutions loaded from {args.solutions}")

    picked = [chs[args.id]] if args.id else list(chs.values())
    dt = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}.get(args.dtype)

    if args.cmd == "list":
        for ch in picked:
            print(f"  {ch.id}  [tier {ch.tier}]  {ch.name}")
            print(f"        {ch.description.splitlines()[0]}")
        return 0

    if args.cmd == "lint":
        import inspect
        from . import lint as lintmod
        total = 0
        for ch in picked:
            mod = inspect.getmodule(ch.solution)
            path = getattr(mod, "__file__", None)
            if not path:
                continue
            findings = lintmod.lint_file(path)
            if findings:
                print(f"  {ch.id} ({path.split('/')[-1]}):")
                for f in findings:
                    print(f"    {f.rule} L{f.line}: {f.message}")
                total += len(findings)
        print(f"\n{total} lint finding(s) across {len(picked)} challenge(s).")
        return 0

    failed = False
    if args.cmd == "test":
        for ch in picked:
            presets = [args.preset] if args.preset else [k for k in ch.presets if k != "bench"]
            for preset in presets:
                for dtype in ([dt] if dt else ch.dtypes):
                    try:
                        r = harness.check(ch, preset, dtype)
                        mark = "PASS" if r.passed else "FAIL"
                        failed |= not r.passed
                        detail = r.detail
                        err = f"max_abs={r.max_abs_err:.2e}"
                    except Exception as e:  # crashing attempt = failed attempt
                        mark, failed, err = "FAIL", True, ""
                        detail = f"{type(e).__name__}: {e}"[:90]
                    print(f"  {mark}  {ch.id} {preset:>8} {str(dtype):>14} {err} {detail}")
        return 1 if failed else 0

    # bench / grade
    results = []
    for ch in picked:
        preset = args.preset or ("bench" if "bench" in ch.presets else next(iter(ch.presets)))
        for dtype in ([dt] if dt else ch.dtypes):
            print(f"  running {ch.id} {preset} {dtype} ...", file=sys.stderr)
            try:
                results.append(harness.bench(ch, preset, dtype))
            except Exception as e:  # crashing attempt = F, keep grading the rest
                print(f"    error: {type(e).__name__}: {e}"[:120], file=sys.stderr)
                results.append(harness.BenchResult(
                    ch.id, preset, str(dtype), float("nan"), float("nan"),
                    0.0, 0.0, 0.0, 0.0, 0.0, "F"))
    print(harness.report(results, path=args.json))
    return 1 if any(r.grade == "F" for r in results) else 0


if __name__ == "__main__":
    sys.exit(main())
