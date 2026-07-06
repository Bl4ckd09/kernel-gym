"""CLI: python -m gym {list|test|bench|grade} [challenge-id] [--json out.json]"""

from __future__ import annotations

import argparse
import sys

import torch

from . import registry
from . import harness


def main() -> int:
    p = argparse.ArgumentParser(prog="gym")
    p.add_argument("cmd", choices=["list", "test", "bench", "grade", "lint"])
    p.add_argument("id", nargs="?", help="challenge id, e.g. t3.02 (default: all)")
    p.add_argument("--preset", default=None, help="input preset (default: all for test, bench for bench)")
    p.add_argument("--dtype", default=None, choices=["fp32", "fp16", "bf16"])
    p.add_argument("--json", default=None, help="write bench results to JSON")
    args = p.parse_args()

    chs = registry.load_all()
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
                    r = harness.check(ch, preset, dtype)
                    mark = "PASS" if r.passed else "FAIL"
                    failed |= not r.passed
                    print(f"  {mark}  {ch.id} {preset:>8} {str(dtype):>14} "
                          f"max_abs={r.max_abs_err:.2e} {r.detail}")
        return 1 if failed else 0

    # bench / grade
    results = []
    for ch in picked:
        preset = args.preset or ("bench" if "bench" in ch.presets else next(iter(ch.presets)))
        for dtype in ([dt] if dt else ch.dtypes):
            print(f"  running {ch.id} {preset} {dtype} ...", file=sys.stderr)
            results.append(harness.bench(ch, preset, dtype))
    print(harness.report(results, path=args.json))
    return 1 if any(r.grade == "F" for r in results) else 0


if __name__ == "__main__":
    sys.exit(main())
