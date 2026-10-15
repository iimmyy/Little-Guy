# 2026-10-15: squeeze pipeline v2, plan (raw notes)

Rule: engineer the model itself, for hardware in general. Universal currency: bytes read per word and total size, in
standard formats any engine reads. Engine/chip tricks and helper models are not part of the method. No
self-distillation healing until compression is done right.

## Problems found in the v1 ruler

- Eval texts are private write-ups: not reproducible by others.
- Probe evals are small (~4 chunks) with no measured noise floor; small "cheap" cuts may be noise.
- KL vs 8-bit is the only measure; no task benchmarks. On E4B, KL looked fine while behavior broke.
- 15-layer blocks: one sensitive layer drags 14 others.
- Azure CPU reads the 31B at ~7 tokens/s: one benchmark pass per model takes hours.

## Steps

A. Ruler: public pinned texts, calibration and test disjoint; noise floor (same file, disjoint halves); KL and same-top
   vs Q8_0 plus task accuracy (HellaSwag, Winogrande, ARC, MMLU from research/big-model/get_bench.sh).
B. Baselines: Q8_0, Google Q4_0, community imatrix Q3_K_M / IQ3_M / Q3_K_S, plain Q3_K_M (tools/squeeze/community_31b.sh).
C. Sensitivity map: 5-layer blocks, sample size chosen from the noise floor.
D. Allocation: mixed precision, standard llama.cpp types only.
E. Rounding: importance-weighted (imatrix), then error-compensating (GPTQ-style) if a gap remains.
F. Verify: same ruler, plus size, bytes per word, speed on the 24 GB Mac.

## Decisions (2026-10-15)

- Public test text: WikiText-2 test from the first article boundary past the midpoint (641,668 chars), disjoint from
  the imatrix calibration (first 80 x 512 tokens). Plus HellaSwag / Winogrande / ARC / MMLU (pinned).
- Private test text: one unpublished paper (provisional patent), Mac only; never leaves the machine, numbers only in
  notes. Unpublished means the model cannot have seen it in training.
- All measurement on the Mac (M4 Pro 24 GB), one engine build, for consistency. Azure only builds files.
- Q8_0 reference (32.6 GB) does not fit in RAM: runs on CPU from disk, once per text.
- Google's Q4_0 does not fit the Mac GPU even with 40 of 60 layers offloaded: measured on CPU (~18 tokens/s reading,
  2048 tokens in 115 s).
- KL reference logits cost 0.5 MB per scored token (262k vocab x 2 bytes): 16 chunks x 1024 scored tokens = 8 GB
  per text.

## v2 design: any model in (2026-10-15)

1. Discover: arch, layer count and every 2D blk tensor from the model's own GGUF (no hand-written group lists).
2. Donors: the whole model at each stock type with the importance matrix, --pure (each tensor exactly that type):
   Q2_K, IQ3_XXS, IQ3_S, IQ4_XS, Q4_K, Q5_K, Q6_K.
3. Predict harm per (tensor, layer, type) without running the model: importance-weighted rounding error
   (sum over columns of imatrix x squared error), calibrated against a few measured KL probes.
4. Allocate per tensor and layer (knapsack on predicted harm vs bytes).
5. Build, measure KL, fix the worst spots, repeat a few rounds.
6. Final: the full ruler on the Mac (public held-out text, private paper, benchmarks), vs the community curve.

Win condition: our quality-vs-size curve above the community curve; headline = the smallest size within the noise
floor of the official 4-bit. Proven on 2-3 model families, not just Gemma.
