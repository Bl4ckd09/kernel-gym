"""Pytest bridge: one correctness test per (challenge, preset, dtype)."""

import pytest
import torch

from gym import registry
from gym import harness

if not torch.cuda.is_available():
    pytest.skip("kernel-gym requires a CUDA GPU", allow_module_level=True)

_CASES = [
    pytest.param(ch, preset, dtype,
                 id=f"{ch.id}-{preset}-{str(dtype).replace('torch.', '')}",
                 marks=getattr(pytest.mark, f"tier{ch.tier}"))
    for ch in registry.load_all().values()
    for preset in ch.presets if preset != "bench"
    for dtype in ch.dtypes
]


@pytest.mark.parametrize("ch,preset,dtype", _CASES)
def test_correctness(ch, preset, dtype):
    r = harness.check(ch, preset, dtype)
    assert r.passed, f"max_abs={r.max_abs_err:.3e} max_rel={r.max_rel_err:.3e} {r.detail}"
