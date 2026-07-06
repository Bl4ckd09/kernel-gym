"""Blank mode: turn the challenge set into a cold-writing eval.

`python -m gym blank --out attempts/stubs` emits one stub per challenge: the module
docstring (the spec), make_inputs and reference (the ground truth), and an empty
`solution()` — with every @jit kernel stripped. A model (or human) fills in the kernel
cold; `python -m gym test/grade --solutions <dir>` then loads their `solution` in place
of the shipped one and grades it with the exact same harness.

Stubs are standalone: they import triton directly instead of gym.tri, so they only ever
run on the GPU box where triton exists.
"""

from __future__ import annotations

import ast
import inspect
import os

_PRELUDE = '''\
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

'''


def _is_kernel_decorator(dec: ast.expr) -> bool:
    name = ""
    if isinstance(dec, ast.Name):
        name = dec.id
    elif isinstance(dec, ast.Attribute):
        name = dec.attr
    elif isinstance(dec, ast.Call):
        return _is_kernel_decorator(dec.func)
    return name in ("jit", "autotune")


def make_stub(module) -> str:
    src = inspect.getsource(module)
    tree = ast.parse(src)
    kept: list[ast.stmt] = []
    for node in tree.body:
        # module docstring
        if (isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant)
                and isinstance(node.value.value, str)):
            kept.append(node)
            continue
        # imports: drop gym imports (prelude replaces gym.tri)
        if isinstance(node, ast.ImportFrom):
            if node.module and node.module.split(".")[0] == "gym":
                continue
            kept.append(node)
            continue
        if isinstance(node, ast.Import):
            kept.append(node)
            continue
        # register(...) call: drop (carries grade thresholds + Challenge wiring)
        if (isinstance(node, ast.Expr) and isinstance(node.value, ast.Call)
                and isinstance(node.value.func, ast.Name)
                and node.value.func.id == "register"):
            continue
        if isinstance(node, ast.FunctionDef):
            if any(_is_kernel_decorator(d) for d in node.decorator_list):
                continue  # strip the shipped kernels — that's the challenge
            if node.name == "solution":
                node.decorator_list = []
                node.body = [
                    ast.Expr(ast.Constant("YOUR KERNEL HERE — see rules at top of file.")),
                    ast.parse("raise NotImplementedError").body[0],
                ]
            kept.append(node)
            continue
        kept.append(node)
    tree.body = kept
    return _PRELUDE + ast.unparse(tree) + "\n"


def emit_stubs(out_dir: str) -> list[str]:
    from . import registry
    os.makedirs(out_dir, exist_ok=True)
    written = []
    for ch in registry.load_all().values():
        module = inspect.getmodule(ch.make_inputs)
        base = os.path.basename(module.__file__)
        path = os.path.join(out_dir, base)
        with open(path, "w") as f:
            f.write(make_stub(module))
        written.append(path)
    return written


def load_solutions(challenges: dict, sol_dir: str) -> None:
    """Patch each challenge's solution from <sol_dir>/<module_basename>.py.

    Strict: a missing or unloadable attempt leaves a solution that raises, so the
    harness records an F rather than silently grading the shipped kernel.
    """
    import importlib.util

    def _fail(msg):
        def f(**kwargs):
            raise NotImplementedError(msg)
        return f

    for ch in challenges.values():
        module = inspect.getmodule(ch.make_inputs)
        base = os.path.basename(module.__file__)
        path = os.path.join(sol_dir, base)
        if not os.path.exists(path):
            ch.solution = _fail(f"no attempt file {path}")
            continue
        try:
            spec = importlib.util.spec_from_file_location(f"attempt_{ch.id.replace('.', '_')}", path)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            ch.solution = mod.solution
        except Exception as e:  # broken attempt = failed attempt
            ch.solution = _fail(f"attempt failed to import: {e!r}")
