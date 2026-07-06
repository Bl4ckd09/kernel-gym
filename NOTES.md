# Extracted kernel ideas — source notes → challenges

Distilled from 505 GPU/CUDA/Triton notes in my second brain (`sources/twitter`). Five
agents read them in parallel; ~305 carried real kernel/perf content. This file is the
durable extraction: what became a challenge, what's queued, and where each came from.

## What became the ladder

| challenge | primary sources | the load-bearing number/insight |
|-----------|-----------------|--------------------------------|
| t1.01 bias+GELU | @cHHillee | fusion removes intermediate HBM traffic; bandwidth is the budget |
| t1.02 SwiGLU | @main_horse, notes on gated MLPs | gated MLPs dominate FFN FLOPs; fuse the activation |
| t2.01 softmax | @cHHillee, @tri_dao | fused softmax keeps the row in SRAM; seed of online softmax |
| t2.02 RMSNorm | @tri_dao | memory-bound row reduction; fuse the scale |
| t2.03 RoPE | @UnslothAI | fused RoPE kernels → 3× training, −30% VRAM |
| t3.01 GEMM | @cHHillee/@gordic_aleksa, @aryanvs_ | square tiles max arithmetic intensity; 75–80% cuBLAS is weeks, last 20% is frontier |
| t3.02 int8 GEMM | @maharshii, @prajdabre | row-scaled int8 3.1–3.5× over bf16; per-row/col scales beat per-tensor on outliers |
| t3.03 LayerNorm fwd | @aryagxr, @MankyDankyBanky | coalesced + shared-mem reductions; fused one-pass 3× over unfused |
| t4.01 fused CE | @danielhanchen, @cHHillee | never materialize (M,V) logits; same online-max as softmax |
| t4.02 LayerNorm bwd | @vllm_project, @cloneofsimo | hand-written matching backward; the backward is the hard, rare part |
| t5.01 Flash fwd | @tri_dao, @pranay5255, @cHHillee | online softmax; FA2's 2–3× is the Q-outer/KV-inner loop order |
| t5.02 Flash bwd | @cloneofsimo, @aryanvs_, @archiexzzz | recompute softmax from stashed logsumexp; differentiable recompute enables double-backward |

## Queued challenges (corroborated across ≥2 note batches, not yet built)

Ranked by how strongly the notes converge on them. Each is a clean next module.

**Tier 3–4**
- ~~**GQA attention**~~ — BUILT as t4.03. (@rohanpaul_ai, @GoSailGlobal, @TheAhmadOsman ×3)
- ~~**W4A16 dequant matmul**~~ — BUILT as t4.04, incl. split-K for decode shapes. (@jeremyphoward, @Yuchenj_UW, @doodlestein, @elliotarledge)
- **Sliding-window causal attention** — each query attends to the last W keys, no full mask materialized. (@karpathy nanochat, @cHHillee, @hamzaelshafie gpt-oss) — window_size kwarg from FA3.
- **FP4 pack/unpack + NVFP4 dequant-matmul** — 2×E2M1 in one uint8; per-block E4M3 scales. (@maharshii ×2, @mobicham, @jackcookjack, @DAlistarh) — Triton fp4 matmul reportedly beat CUTLASS.
- **Fused SwiGLU MLP** — gate·SiLU·up then down, intermediates never hit HBM. (@YouJiacheng/@anneouyang "120% of PyTorch", @HeMuyu0327 multi-head FFN) — torch.compile can't fuse matmul+silu+matmul.
- **Top-k selection kernel** — bitonic for small k; expert routing / sampling. (@AlpinDale, @anaumghori, SonicMoE bitonic top-k up to 20–30× torch.topk)
- **Fused RMSNorm + residual add** (and its exact backward) — the vLLM train/infer-parity kernel. (@vllm_project, @TheAhmadOsman)
- **INT8 / INT4 KV-cache** — quantize-on-write, dequant-on-read inside attention, per-token scales. (@_reesechong 3.78× smaller cache, @TraffAlex q4_0 KV, @labubu_trader ~10×)
- **Coalesced tiled transpose** — shared-mem staging + `+1` padding to kill bank conflicts. (@jino_rohit, @RubenVeidt)

**Tier 4–5**
- **Batch-invariant / deterministic split-K reduction** — fixed reduction order regardless of grid/batch split; bitwise-reproducible. (@thinkymachines, @gabriberton, @SemiAnalysis, @HeMuyu0327) — the nondeterminism source is batch-size-dependent reduction order.
- **Split-K GEMM** — partial-sum reduction across K-splits for skinny-K shapes. (@leloykun Lean4→TileLang rediscovered it)
- **Grouped / pooled expert GEMM (MoE)** — gather top-k routed tokens per expert into contiguous batches, matmul, scatter back. (@tri_dao SonicMoE 1.86×, @_xjdr RDEP, @UnslothAI 12× MoE, @PatrickToulme)
- **Paged-KV decode attention** — gather K/V from a non-contiguous block table. (@techwith_ram, @remi_or_, @RedHat_AI, @h100envy FlashInfer)
- **Muon Newton-Schulz orthogonalization** — 5-step polar-factor iteration of the grad matrix, one fused kernel. (@archiexzzz, @kellerjordan0) — plus fused QK-RMSNorm (diff 2) and tanh logit softcap (diff 1).
- **MLA decode attention** — expand compressed latent KV, RoPE/NoPE split. (@TheAhmadOsman, @levidiamode, @teortaxesTex) — KV cache, not weights, is the memory tax.
- **Gated-DeltaNet / linear-attention chunkwise scan** — O(1) recurrent state, WY-representation forward + gate-aware backward. (@zhuokaiz Qwen3.5, @Kimi_Moonshot KDA, @rasbt GDN-2) — 75% of layers hold zero KV cache.
- **Decode megakernel** — fuse RMSNorm→QKV→attn→O→MLP for batch-1 into one persistent kernel. (@AlpinDale, @emre570_, @elliotarledge 18.7×)

## Cross-cutting principles the notes hammer (baked into the harness)

- **Roofline / "move less data".** FLOPS outpace bandwidth, so most kernels are
  bandwidth-bound; fusion is the lever. Decode is memory-bound (~200–300 tok/s ceiling),
  prefill is compute-bound. → harness reports arithmetic intensity + % of empirical roofline.
- **fp32 accumulation.** tf32 carries ~19-bit mantissa; Hopper fp8 wgmma accumulates in
  ~fp22. Accumulate reductions/matmuls in fp32 even for fp16/bf16 data. → lint rule KG003.
- **`.contiguous()` footgun.** Non-contiguous strides silently corrupt kernel output/grads
  (@giffmana). → lint rule KG004 (it caught my own FlashAttention fwd).
- **Measure kernels correctly.** CUDA events on the wrong stream collapsed a "1000×" to
  ~2× (@aryanvs_). → harness uses `triton.testing.do_bench` (warmup + median).
- **Grader adversarialism.** Models have topped kernel competitions by editing grader
  tolerances / reward-hacking (@marksaroufim, KERNELGYM/Dr.Kernel). → correctness is a hard
  gate with fixed per-dtype tolerances and independent CPU reference cross-checks; the
  benchmark can't lift an F.

## External benchmark targets named in the notes

KernelBench (CUDA-L1: 17.7× avg / 449× max on A100), GPUMode/KernelBot ($100K comps, 400K
submissions), Tensara & LeetGPU (free H100/A100), NVIDIA SOL-ExecBench, Anthropic's
cycle-count kernel challenge (147,734 → 1,363 cycles). Useful for calibrating grade
thresholds and as a next-level harness to plug into.
