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

## Open decisions

- Test text: held-out tail of WikiText-2 test (already pinned, calibration uses the first ~41k tokens) vs a public
  chat dataset (new download).
- Compute: Azure CPU (slow, hours per benchmark) vs a GPU VM for measurement.
