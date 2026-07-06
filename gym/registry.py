"""Challenge registry: every challenge module in challenges/ self-registers here."""

from __future__ import annotations

import importlib
import pkgutil
from dataclasses import dataclass, field
from typing import Callable

import torch


@dataclass
class Challenge:
    id: str                     # e.g. "t3.02"
    name: str                   # e.g. "row softmax (online)"
    tier: int                   # 1..5
    description: str            # what to build and why it matters
    sources: list[str]          # note citations: "@handle — takeaway"
    make_inputs: Callable       # (preset: str, device, dtype) -> dict of tensors/args
    reference: Callable         # **inputs -> Tensor; eager PyTorch ground truth
    solution: Callable          # **inputs -> Tensor; the Triton implementation
    flops: Callable             # (inputs: dict) -> float, useful FLOPs for the preset
    bytes: Callable             # (inputs: dict) -> float, min HBM traffic in bytes
    baseline: Callable | None = None      # benchmark opponent; defaults to reference
    presets: dict = field(default_factory=lambda: {})
    dtypes: tuple = (torch.float32, torch.float16)
    tol: dict = field(default_factory=lambda: {
        torch.float32: (1e-4, 1e-5),
        torch.float16: (1e-2, 1e-3),
        torch.bfloat16: (2e-2, 2e-2),
    })
    # speedup vs baseline required for each grade (correctness alone = C)
    grade_b: float = 0.80
    grade_a: float = 1.00

    def __post_init__(self):
        if self.baseline is None:
            self.baseline = self.reference


_REGISTRY: dict[str, Challenge] = {}


def register(ch: Challenge) -> Challenge:
    if ch.id in _REGISTRY:
        raise ValueError(f"duplicate challenge id {ch.id}")
    _REGISTRY[ch.id] = ch
    return ch


def load_all() -> dict[str, Challenge]:
    """Import every module under challenges/ so registrations run."""
    import challenges
    for m in pkgutil.iter_modules(challenges.__path__):
        importlib.import_module(f"challenges.{m.name}")
    return dict(sorted(_REGISTRY.items()))


def get(cid: str) -> Challenge:
    load_all()
    if cid not in _REGISTRY:
        raise KeyError(f"unknown challenge {cid!r}; run `python -m gym list`")
    return _REGISTRY[cid]
