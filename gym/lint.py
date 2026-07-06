"""Static kernel-quality lint — a code-smell pass over challenge solutions.

Adapted from the user's `triton-kernel-lint` repo (TKL rules): it parses the Python AST
of a challenge module and flags perf anti-patterns *before* runtime. In kernel-gym it is a
grading signal orthogonal to the benchmark: a kernel can be fast and still trip a lint (or
pass lint and be slow). Reported by `python -m gym lint <id>`.

Rules:
  KG001  suspicious block size    — BLOCK_* that is not a power of two, or > 8192
  KG002  unusual num_warps        — not in {1,2,4,8,16}
  KG003  fp16/bf16 accumulate     — tl.zeros/tl.dot accumulator not float32/int32
  KG004  missing .contiguous()    — solution() indexes strides but never calls .contiguous()
                                     (@giffmana's silent non-contiguous-gradient footgun)
"""

from __future__ import annotations

import ast
from dataclasses import dataclass


@dataclass
class Finding:
    rule: str
    line: int
    message: str


_POW2 = lambda v: isinstance(v, int) and v > 0 and (v & (v - 1)) == 0


class _Rules(ast.NodeVisitor):
    def __init__(self, src: str):
        self.src = src
        self.findings: list[Finding] = []
        self._solution_indexes_strides = False
        self._solution_has_contiguous = "contiguous" in src

    def visit_Assign(self, node: ast.Assign):
        for t in node.targets:
            if isinstance(t, ast.Name) and t.id.startswith("BLOCK"):
                v = getattr(node.value, "value", None)
                if isinstance(v, int) and not (_POW2(v) and v <= 8192):
                    self.findings.append(Finding(
                        "KG001", node.lineno,
                        f"{t.id}={v} is not a power-of-two ≤ 8192 (poor coalescing / occupancy)"))
        self.generic_visit(node)

    def visit_keyword(self, node: ast.keyword):
        if node.arg == "num_warps":
            v = getattr(node.value, "value", None)
            if isinstance(v, int) and v not in (1, 2, 4, 8, 16):
                self.findings.append(Finding(
                    "KG002", node.value.lineno,
                    f"num_warps={v} is unusual; use one of 1,2,4,8,16"))
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call):
        fn = _dotted(node.func)
        if fn in ("tl.zeros",) and node.args:
            dt = _find_kw(node, "dtype")
            if dt and "float16" in dt or dt and "bfloat16" in dt:
                self.findings.append(Finding(
                    "KG003", node.lineno,
                    "accumulator dtype is fp16/bf16; accumulate in float32 for numerical safety"))
        if fn == "tl.dot":
            od = _find_kw(node, "out_dtype")
            if od and ("float16" in od or "bfloat16" in od):
                self.findings.append(Finding(
                    "KG003", node.lineno,
                    "tl.dot out_dtype is fp16/bf16; use float32 accumulation"))
        self.generic_visit(node)


def _dotted(node) -> str:
    if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
        return f"{node.value.id}.{node.attr}"
    if isinstance(node, ast.Name):
        return node.id
    return ""


def _find_kw(call: ast.Call, name: str) -> str | None:
    for kw in call.keywords:
        if kw.arg == name:
            return ast.unparse(kw.value)
    return None


def _uses_strides_without_contiguous(tree: ast.AST, src: str) -> bool:
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "solution":
            body = ast.get_source_segment(src, node) or ""
            uses_stride = ".stride(" in body
            has_contig = ".contiguous()" in body
            return uses_stride and not has_contig
    return False


def lint_source(src: str) -> list[Finding]:
    tree = ast.parse(src)
    r = _Rules(src)
    r.visit(tree)
    if _uses_strides_without_contiguous(tree, src):
        r.findings.append(Finding(
            "KG004", 1,
            "solution() passes .stride() to a kernel but never calls .contiguous(); "
            "non-contiguous inputs silently corrupt results (@giffmana footgun)"))
    return sorted(r.findings, key=lambda f: (f.line, f.rule))


def lint_file(path: str) -> list[Finding]:
    with open(path) as f:
        return lint_source(f.read())
