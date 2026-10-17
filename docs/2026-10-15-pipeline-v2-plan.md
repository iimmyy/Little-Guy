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

## Next models (generality)

- Model #2: Gemma 4 26B-A4B (mixture of experts, ~4B active of 26B). ROCK 5C (RK3588S2, 32 GB) runs Unsloth's
  QAT UD-Q4_K_XL (14.2 GB) at 7.00 tok/s generation, 15.9 prompt (forum.radxa.com/t/31145); E4B int-4 at 8.81,
  31B int-4 at 1.42 (forum.radxa.com/t/31241). Q6A has 12 GB (2 x 6 GB LPDDR5): the 26B-A4B needs ~9-10 GB to fit.
  New lever for MoE: per-expert allocation (rarely routed experts tolerate fewer bits).

## v3 order (Immy, 2026-10-15): match first, then allocate

Output matching first, allocation second. For every matrix and every candidate type: round with error compensation
(GPTQ-style, using the input covariance from calibration text) so the matrix's OUTPUT matches the original, then
measure the residual output error. Allocate bits on that residual, not on plain rounding error: a matrix that looks bad
under plain rounding may match well after compensation and never need the extra bit. Each extra bit doubles the levels
(2-bit 4, 3-bit 8, 4-bit 16) for ~+33% bytes on that matrix (3 -> 4), so it is spent only where matching falls short.
Needs: full input covariance per matrix (an imatrix extension), a compensating quantizer that writes stock types.

## Stage B result 1: output matching on E4B, 3-bit (2026-10-15)

Same ruler for all rows: WikiText-2 test second half (held out from calibration), 32 x 512 chunks, KL vs E4B Q8_0,
Mac GPU. Inputs recorded on the first 16,384 tokens of the calibration text (LG_ACT_DUMP), importance matrix on 80 x 512.

| E4B | Bytes | KL | Same top |
|---|---|---|---|
| Q3_K --pure + importance matrix (stock rounding) | 3,267,811,488 | 0.261 | 77.2% |
| same file, codes re-picked by output matching | 3,267,811,488 | 0.156 | 82.4% |
| Google QAT Q4_0 (reference) | 5,154,940,832 | 0.033 | 91.3% |

- Drift -40% at an identical size; only ~15% of codes changed. Per-matrix output error -71% in-sample, -19..-88%
  held-out (ffn_down weakest: 10,240 inputs, ~12k recorded rows).
- Matching runs ~33 min for the whole E4B on the Mac CPU (numpy).

## Correction (2026-10-15, late): output matching overfits the calibration text

Same files, every E4B ruler (persona = private persona chat; chat / raw = write-up texts; wiki = public held-out half):

| E4B | Per word | Persona | Chat | Raw | Wiki |
|---|---|---|---|---|---|
| mix-21 (live) | 1.78 GB | 0.482 | 0.441 | 0.491 | 0.447 |
| mix-21, matched on WikiText | 1.78 GB | 0.547 | 0.474 | 0.577 | 0.316 |
| Q3_K + importance matrix | 2.04 GB | 0.341 | 0.269 | 0.325 | 0.261 |
| Q3_K, matched on WikiText | 2.04 GB | 0.354 | 0.288 | 0.323 | 0.156 |

- Matching on WikiText inputs only helps WikiText-like inputs; on chat it ties (3-bit) or hurts (2-bit, 4 levels: each
  re-picked code is a big jump). The earlier "-40%" is in-domain only. mix-21 matched does NOT go live.
- Plain Q3_K beats the live mix-21 on every ruler by 29-42% for ~15% more bytes per word.
- Next: calibrate on representative text (model-generated conversations, none of the eval texts), stronger damping.

## 2026-10-16 night: representative calibration, 3-bit NPU kernel, Little Guy on the board

Calibration text: 240 self-generated chat exchanges (E4B Q8 answering everyday prompts, tools/squeeze/make_selfchat.py)
interleaved with the WikiText calibration half (.run/bench/calib_mix.txt), chat markers as control tokens.
Held-out chat ruler: 160 exchanges on different topics and request types (make_selfchat.py eval).

E4B, KL vs Q8_0 (lower is better), bytes read per generated word:

| E4B | Per word | Persona | Chat | Raw | Wiki | Held-out chat |
|---|---|---|---|---|---|---|
| mix-21 (live) | 1.78 | 0.482 | 0.441 | 0.491 | 0.447 | 0.391 |
| Q3_K, mixed imatrix | 2.04 | 0.300 | 0.269 | 0.305 | 0.260 | 0.213 |
| Q3_K, mixed imatrix, matched (damping 0.1) | 2.04 | 0.167 | 0.213 | 0.260 | 0.173 | 0.144 |
| community IQ3_XXS (same imatrix) | 2.03 | 0.372 | 0.310 | 0.362 | 0.337 | 0.290 |
| community Q2_K | 2.05 | 0.426 | 0.437 | 0.521 | 0.437 | 0.391 |
| community Q3_K_S | 2.30 | 0.282 | 0.249 | 0.289 | 0.242 | 0.203 |
| community IQ3_M | 2.36 | 0.241 | 0.178 | 0.196 | 0.173 | 0.155 |
| community Q3_K_M | 2.50 | 0.152 | 0.115 | 0.133 | 0.114 | 0.095 |
| community Q4_K_S | 2.84 | 0.044 | 0.022 | 0.024 | 0.023 | 0.019 |
| Google QAT Q4_0 | 2.83 | 0.039 | 0.029 | 0.033 | 0.033 | 0.027 |

- Damping 0.1 generalizes better than 0.01 (persona 0.167 vs 0.254). Matched Q3_K at 2.04 GB/word ~ community IQ3_M at
  2.36 (14% fewer bytes); half the drift of the same-size community quant.

Board (Q6A NPU, engine ~/preview3 with the Q3_K kernel; NPU vs board CPU KL 0.004 on Q3_K):

| 80 tries (seeds 1000, 5000) | mix-21 | Q3_K plain (wiki imatrix) | Q3_K matched (mixed) |
|---|---|---|---|
| Dayvigo warns first | 57 | 57 | 78 |
| Dayvigo dodges first | 13 | 9 | 0 |
| dog firm no | 77 | 78 | 80 |
| pasta "Hm" | 25 | 16 | 6 |
| invents his day (200 tries) | 7 | 5 (of 80) | 26 |
| chat speed, NPU (words/s) | 10.2 | 8.65 | 8.53 |

- Matched Q3_K invents his day 13% vs mix-21 3.5%: does not go live (rule: tie or beat every check).
- Open: the 262,144-row dictionary is wrong on the NPU Q3_K kernel (kept off; the preview is Q2_K).
- Original Google QAT Q4_0 on the same board test: invents his day 29/200 (14.5%), matched Q3_K 26/200 (13%), mix-21
  7/200 (3.5%). The matched brain is faithful to the original; mix-21's low rate is a side effect of its distortion.
  Inventing his day is the base model's habit (persona / fine-tuning). Kept mix-21 live per the agreed rule; matched
  Q3_K (e4b-q3k-mm-preview) is on the board for Immy's call. Original Q4_0 talks at 6.6 words/s on the NPU.

## 2026-10-16 morning: near-original E4B, behavior-checker audit, reset

- Near-original E4B (v3 at 2.09 GB matrices, 2.40 GB/word with preview): persona KL 0.047, chat 0.045, held-out 0.030
  vs Google QAT Q4_0 0.039 / 0.029 / 0.027; board NPU 7.40 tok/s vs Google 6.0 (mix-21 9.27). v3 at 2.19 GB/word:
  persona 0.090, 7.9 tok/s.
- 31B Google Q4_0 on the Mac rulers: wiki 0.089, held-out chat 0.049 (ours 13.21 GB: 0.278 / 0.190).
- Behavior checker audit (replies read one by one): keyword checks miss "I'm not sure" dodges, count wrong "it's okay"
  answers as warnings, undercount invented days without persona (5 flagged vs ~33 by reading), and cannot see invented
  facts. Behavior verdicts from 2026-10-15/16 need re-deciding by reading. Replies are now saved per run
  (.run/results/behavior/), survivors archived.
- Reset: pause board runs, write a behavior rubric, one clean head-to-head read in full, then decide.

## 2026-10-16 afternoon/evening: fresh E4B rebuild (raw, no persona, measured against Google's QAT 16-bit)

- Started over from Google's release: checksums match Google's published ones; our 16-bit copy is bit-identical (665/665).
- Audit fixes: old matching saw only the first 16,384 input rows (1.6 rows per input on ffn_down); now every row (~15 per
  input), held-out = whole documents; bit-mix probes had been tuned on the WikiText ruler, now on their own dev text.
- Google's 4-bit = plain llama.cpp Q4_0 rounding of their QAT weights (every block matrix byte-identical), Q6_K word tables.
- The scale plain rounding picks sits ~0.1% off the grid QAT trained; an importance-weighted scale search lands on it.
  Same bytes (Google's file, only block scales changed): drift 0.00033 / 0.00016 vs Google 0.0370 / 0.0274 (110-170x less),
  top word 99.3% vs 91.0%, perplexity 32.38 vs 33.90 (16-bit 32.43). Benchmarks (log-likelihood, thinking off) tie with
  the 16-bit: HellaSwag 68.72 vs 68.73, MMLU 36.50 vs 36.50, Winogrande 67.80 vs 67.64. Board NPU speed equal (6.60 vs 6.52).
- Behavior, every reply read: the fixed file reproduces the 16-bit's choices (15/160 identical replies; Google's 4-bit 0/160).
  Google's own model invents its day, dodges Dayvigo and invents what it is; those are not compression damage.
- Next: a ladder of single changes from the fixed file (dictionary preview, lossless packing, matched 3/2-bit, training-free
  weight edits), each checked against the 16-bit, toward faster while still matching it.
