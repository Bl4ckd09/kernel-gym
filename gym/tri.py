"""Triton import shim.

Triton has no macOS/CPU backend, but we still want challenge modules to *import*
on a laptop so `gym list`, the reference-logic tests, and the linter all run
without a GPU. When Triton is missing we install passthrough stubs: `@jit`
returns the function untouched, and `tl`/`triton` attribute access yields a inert
object usable in annotations like `BLOCK: tl.constexpr`. The kernels are only ever
*called* on CUDA, where the real Triton is present, so nothing is lost.
"""

from __future__ import annotations

try:
    import triton  # type: ignore
    import triton.language as tl  # type: ignore

    HAS_TRITON = True
    jit = triton.jit
    autotune = triton.autotune
    Config = triton.Config
    cdiv = triton.cdiv

except ImportError:  # laptop / CI without CUDA
    HAS_TRITON = False

    class _Inert:
        """Absorbs any attribute access, call, or subscription."""
        def __getattr__(self, _):
            return self
        def __call__(self, *a, **k):
            return self
        def __getitem__(self, _):
            return self

    triton = _Inert()  # type: ignore
    tl = _Inert()      # type: ignore

    def jit(fn=None, **_):
        return fn if fn is not None else (lambda f: f)

    def autotune(*_a, **_k):
        return lambda f: f

    def Config(*_a, **_k):  # noqa: N802 (mirror triton.Config)
        return None

    def cdiv(a, b):
        return -(-a // b)


def require_triton():
    if not HAS_TRITON:
        raise SystemExit(
            "This kernel needs Triton + a CUDA GPU. Sync kernel-gym to a GPU box "
            "(see SETUP.md) and rerun there."
        )
