"""CPU verification of the reference implementations (the ground truth).

Triton kernels can't run without CUDA, but the pure-PyTorch reference() of each challenge
CAN run on CPU. These tests check each reference against a second, independently written
naive computation. If a reference is wrong, every GPU grade built on it would be wrong —
so this is the load-bearing local check. Runs anywhere, no GPU needed.
"""

import math

import pytest
import torch
import torch.nn.functional as F

from gym import registry

DEV = torch.device("cpu")
torch.manual_seed(0)


def _ch(cid):
    return registry.get(cid)


def test_t1_01_bias_gelu():
    ch = _ch("t1.01")
    inp = ch.make_inputs("small", DEV, torch.float32)
    ref = ch.reference(**inp)
    v = inp["x"] + inp["bias"]
    naive = 0.5 * v * (1 + torch.tanh(math.sqrt(2 / math.pi) * (v + 0.044715 * v**3)))
    assert torch.allclose(ref, naive, atol=1e-5)


def test_t1_02_swiglu():
    ch = _ch("t1.02")
    inp = ch.make_inputs("small", DEV, torch.float32)
    ref = ch.reference(**inp)
    a = inp["a"]
    naive = (a * torch.sigmoid(a)) * inp["b"]
    assert torch.allclose(ref, naive, atol=1e-5)


def test_t2_01_softmax():
    ch = _ch("t2.01")
    inp = ch.make_inputs("small", DEV, torch.float32)
    ref = ch.reference(**inp)
    x = inp["x"]
    e = torch.exp(x - x.max(-1, keepdim=True).values)
    naive = e / e.sum(-1, keepdim=True)
    assert torch.allclose(ref, naive, atol=1e-6)
    assert torch.allclose(ref.sum(-1), torch.ones(x.shape[0]), atol=1e-5)


def test_t2_02_rmsnorm():
    ch = _ch("t2.02")
    inp = ch.make_inputs("small", DEV, torch.float32)
    ref = ch.reference(**inp)
    x, w, eps = inp["x"], inp["weight"], inp["eps"]
    naive = x / torch.sqrt((x**2).mean(-1, keepdim=True) + eps) * w
    assert torch.allclose(ref, naive, atol=1e-5)


def test_t2_03_rope():
    ch = _ch("t2.03")
    inp = ch.make_inputs("small", DEV, torch.float32)
    ref = ch.reference(**inp)
    # independent rotate-half: build complex rotation
    x, cos, sin = inp["x"], inp["cos"], inp["sin"]
    B, S, H, D = x.shape
    half = D // 2
    x1, x2 = x[..., :half], x[..., half:]
    c, s = cos[..., :half].unsqueeze(2), sin[..., :half].unsqueeze(2)
    naive = torch.cat([x1 * c - x2 * s, x2 * c + x1 * s], dim=-1)
    assert torch.allclose(ref, naive, atol=1e-5)


def test_t3_01_matmul():
    ch = _ch("t3.01")
    inp = ch.make_inputs("square", DEV, torch.float32)
    ref = ch.reference(**inp)
    naive = torch.einsum("mk,kn->mn", inp["a"], inp["b"])
    assert torch.allclose(ref, naive, atol=1e-3, rtol=1e-3)


def test_t3_02_int8_matmul():
    ch = _ch("t3.02")
    inp = ch.make_inputs("square", DEV, torch.float32)
    ref = ch.reference(**inp)
    xq, wq, xs, ws = inp["xq"], inp["wq"], inp["xs"], inp["ws"]
    acc = (xq.to(torch.int32).float() @ wq.to(torch.int32).float())
    naive = (acc * xs[:, None] * ws[None, :]).half()
    assert torch.allclose(ref.float(), naive.float(), atol=1e-2, rtol=1e-2)


def test_t3_03_layernorm():
    ch = _ch("t3.03")
    inp = ch.make_inputs("small", DEV, torch.float32)
    ref, mean, rstd = ch.reference(**inp, return_stats=True)
    x, w, b, eps = inp["x"], inp["weight"], inp["bias"], inp["eps"]
    mu = x.mean(-1, keepdim=True)
    var = x.var(-1, unbiased=False, keepdim=True)
    naive = (x - mu) / torch.sqrt(var + eps) * w + b
    assert torch.allclose(ref, naive, atol=1e-5)
    assert torch.allclose(mean, mu.squeeze(-1), atol=1e-5)
    assert torch.allclose(rstd, torch.rsqrt(var + eps).squeeze(-1), atol=1e-5)


def test_t4_01_cross_entropy():
    ch = _ch("t4.01")
    inp = ch.make_inputs("small", DEV, torch.float32)
    ref = ch.reference(**inp)
    logits, target = inp["logits"], inp["target"]
    lse = torch.logsumexp(logits, dim=-1)
    per_row = lse - logits.gather(1, target[:, None]).squeeze(1)
    assert torch.allclose(ref, per_row.mean(), atol=1e-4)


def test_t4_02_layernorm_backward():
    ch = _ch("t4.02")
    inp = ch.make_inputs("small", DEV, torch.float32)
    dx, dw, db = ch.reference(**inp)
    # independent autograd path with bias
    x = inp["x"].clone().float().requires_grad_(True)
    w = inp["weight"].clone().float().requires_grad_(True)
    b = torch.zeros_like(w, requires_grad=True)
    y = F.layer_norm(x, (x.shape[-1],), w, b, eps=1e-5)
    y.backward(inp["dy"].float())
    assert torch.allclose(dx.float(), x.grad, atol=1e-4)
    assert torch.allclose(dw.float(), w.grad, atol=1e-3)
    assert torch.allclose(db.float(), b.grad, atol=1e-3)


def test_t4_03_gqa():
    ch = _ch("t4.03")
    inp = ch.make_inputs("small", DEV, torch.float32)
    ref = ch.reference(**inp)
    q, k, v = inp["q"], inp["k"], inp["v"]
    Z, Hq, N, D = q.shape
    G = Hq // k.shape[1]
    naive = torch.empty_like(q)
    mask = torch.tril(torch.ones(N, N, dtype=torch.bool))
    for h in range(Hq):
        s = (q[:, h] @ k[:, h // G].transpose(-1, -2)) / math.sqrt(D)
        s = s.masked_fill(~mask, float("-inf"))
        naive[:, h] = torch.softmax(s, dim=-1) @ v[:, h // G]
    assert torch.allclose(ref, naive, atol=1e-4)


def test_t4_04_w4a16():
    ch = _ch("t4.04")
    inp = ch.make_inputs("small", DEV, torch.float32)
    ref = ch.reference(**inp)
    x, wp, sc = inp["x"], inp["w_packed"], inp["scales"]
    N, Kp = wp.shape
    K = 2 * Kp
    # independent unpack: nibble n at column k comes from byte k//2, low if k even
    w = torch.empty(N, K)
    w[:, 0::2] = (wp & 0xF).float() - 8.0
    w[:, 1::2] = ((wp >> 4) & 0xF).float() - 8.0
    from challenges.t4_w4a16_matmul import GROUP
    w = (w * sc.float().repeat_interleave(GROUP, dim=1)).half()
    naive = (x.float() @ w.float().T).half()
    assert torch.allclose(ref.float(), naive.float(), atol=1e-3)
    # baseline w_fp16 must equal the dequantized weights
    assert torch.equal(inp["w_fp16"], w)


def test_t4_05_sliding_window():
    ch = _ch("t4.05")
    inp = ch.make_inputs("small", DEV, torch.float32)
    ref = ch.reference(**inp)
    q, k, v, W = inp["q"], inp["k"], inp["v"], inp["window"]
    Z, H, N, D = q.shape
    i = torch.arange(N)
    band = (i[:, None] >= i[None, :]) & (i[:, None] - i[None, :] < W)
    s = (q @ k.transpose(-1, -2)) / math.sqrt(D)
    s = s.masked_fill(~band, float("-inf"))
    naive = torch.softmax(s, dim=-1) @ v
    assert torch.allclose(ref, naive, atol=1e-4)
    # a query beyond the window must NOT see the earliest key (sanity on the band)
    assert W < N


def test_t5_03_moe_grouped_gemm():
    ch = _ch("t5.03")
    inp = ch.make_inputs("small", DEV, torch.float32)
    ref = ch.reference(**inp)
    x, w, g = inp["sorted_x"], inp["weights"], inp["group_sizes"]
    assert int(g.sum()) == x.shape[0]
    off = 0
    for e in range(w.shape[0]):
        n = int(g[e])
        if n:
            block = x[off:off + n] @ w[e]
            assert torch.allclose(ref[off:off + n], block, atol=1e-3, rtol=1e-3)
        off += n
    # tile schedule must cover exactly the right number of m-tiles
    from challenges.t5_moe_grouped_gemm import _tile_schedule, BLOCK_M
    te, tm, total = _tile_schedule(g)
    assert total == int(((g + BLOCK_M - 1) // BLOCK_M).sum())
    assert te.shape[0] == total and tm.shape[0] == total


def test_t5_01_flash_forward():
    ch = _ch("t5.01")
    inp = ch.make_inputs("small", DEV, torch.float32)
    ref = ch.reference(**inp)
    q, k, v = inp["q"], inp["k"], inp["v"]
    Z, H, N, D = q.shape
    s = (q @ k.transpose(-1, -2)) / math.sqrt(D)
    mask = torch.tril(torch.ones(N, N, dtype=torch.bool))
    s = s.masked_fill(~mask, float("-inf"))
    p = torch.softmax(s, dim=-1)
    naive = p @ v
    assert torch.allclose(ref, naive, atol=1e-4)


def test_t5_02_flash_backward():
    ch = _ch("t5.02")
    inp = ch.make_inputs("small", DEV, torch.float32)
    dq, dk, dv = ch.reference(**inp)
    q = inp["q"].clone().float().requires_grad_(True)
    k = inp["k"].clone().float().requires_grad_(True)
    v = inp["v"].clone().float().requires_grad_(True)
    N = q.shape[2]
    s = (q @ k.transpose(-1, -2)) * inp["sm_scale"]
    mask = torch.tril(torch.ones(N, N, dtype=torch.bool))
    s = s.masked_fill(~mask, float("-inf"))
    o = torch.softmax(s, dim=-1) @ v
    o.backward(inp["do"].float())
    assert torch.allclose(dq.float(), q.grad, atol=1e-3, rtol=1e-3)
    assert torch.allclose(dk.float(), k.grad, atol=1e-3, rtol=1e-3)
    assert torch.allclose(dv.float(), v.grad, atol=1e-3, rtol=1e-3)


def test_flash_bwd_inputs_are_self_consistent():
    """The stashed L and delta must match the forward that produced o."""
    ch = _ch("t5.02")
    inp = ch.make_inputs("small", DEV, torch.float32)
    q, k, v, o, do = inp["q"], inp["k"], inp["v"], inp["o"], inp["do"]
    N, D = q.shape[2], q.shape[3]
    s = (q.float() @ k.float().transpose(-1, -2)) * inp["sm_scale"]
    mask = torch.tril(torch.ones(N, N, dtype=torch.bool))
    s = s.masked_fill(~mask, float("-inf"))
    L = torch.logsumexp(s, dim=-1)
    o2 = torch.softmax(s, dim=-1) @ v.float()
    assert torch.allclose(inp["L"], L, atol=1e-4)
    assert torch.allclose(o.float(), o2, atol=1e-4)
    assert torch.allclose(inp["delta"], (o.float() * do.float()).sum(-1), atol=1e-4)
