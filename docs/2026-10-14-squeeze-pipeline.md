# 2026-10-14 (evening): from one squeezed brain to a squeeze pipeline

Raw notes and numbers. Code: tools/squeeze, research/big-model, engine/patches (fork 894bb27 + 32 patches).

## Goal

Take a model Google trained for 4-bit (QAT), squeeze it further with Little Guy's tricks, and make it run well on
consumer hardware (24 GB Mac, no swap) while staying as smart as Google's own 4-bit. Make it a pipeline anyone can
rerun: pinned versions, public test files, one command per step.

## Gemma 4 31B, first try with Little Guy's recipe (brain C)

- Built on Azure (D16as_v4, 256 GB scratch disk): master 62.6 GB downloaded in ~7 min (home internet: ~2 h).
- Converter bug: transformers 5.x drops global_head_dim from Gemma 4's config; transformers 4.57 cannot load the
  tokenizer. Fix: research/big-model/convert_wrap.py restores missing settings from config.json.
- Result: 13.86 GB vs Google's 4-bit 17.65 GB (21% smaller). Google's 4-bit does not fit a 24 GB Mac's GPU
  (out of memory at the default limit); ours does.

| 31B, vs 8-bit, raw text | KL | Same top word |
|---|---|---|
| Google official 4-bit | 0.097 | 88.4% |
| Brain C recipe | 0.767 | 62.8% |

- Raw-text perplexity of 31B-it is huge in llama.cpp itself (Google's file 379 on our text, 1,706 on WikiText-2;
  upstream llama.cpp without our patches gives the same 1,705.93). Others report the same (dotLLM PR #788). Not a
  bug: the instruction-tuned 31B is far out of distribution on raw text. Only relative comparisons are meaningful.

## Why 31B is extra sensitive

Solo tests (one group back to 4-bit from Google's file):

| Back at 4-bit | Extra | KL (from 0.767) |
|---|---|---|
| attn_q | ~0.7 GB | 0.748 |
| attn_k | ~0.27 GB | 0.564 |
| ffn_gate | ~1.6 GB | 0.535 |
| ffn_up | ~1.6 GB | 0.653 |

31B has attention_k_eq_v: its 10 global layers have no V matrix, K is used as V. Squeezing K squeezes V there too.
E4B does not have this, so its recipe was safe on E4B but not on 31B.

## E4B for comparison (same method, its own 8-bit)

| E4B | KL raw | Same top |
|---|---|---|
| Google 4-bit | 0.033 | 91.6% |
| brain C | 0.406 | 72.1% |
| mix-21 (live brain) | 0.491 | 69.3% |

31B drops ~26 points of same-top with the recipe, E4B ~20.

## The pipeline (tools/squeeze)

donors (whole model at each bit level) -> probe (one group x layer block upgraded at a time, scored by KL vs 8-bit)
-> allocate / refine (spend a size budget where it buys the most) -> build (copy tensors from donors) -> score.

- Board target (E4B on the NPU): only Q2_K and Q4_0 allowed (the only NPU kernels). Mac target (31B): 3-bit too.
- First allocation assumed gains add up: predicted KL 0.17, measured 0.41. Fixes overlap. The refine step
  re-measures each round (batch greedy). E4B at 4.66 GB: allocate 0.412, refine 0.399, mix-21 0.441 (chat text).
- E4B probes: the middle layers (14-27) are the sensitive ones (attn_v, attn_output, ffn_up); first and last layers
  barely care.

## The lesson of the day: generic drift can lie about behavior

The refined E4B mix (lower drift than mix-21 on generic text) failed his behavior test on the NPU:

| 40 tries, same seeds | mix-21 | pipeline mix |
|---|---|---|
| Dayvigo warns first | 26 | 24 |
| Dayvigo dodges first | 6 | 10 |
| Invents his day | 0 | 7 |
| Dog firm no | 39 | 40 |

Fix: a persona eval. Gold replies from the 4-bit QAT brain to 60 new prompts in the same spirit (never the test
questions; one grapefruit prompt removed for being too close), one chat transcript with the persona, scored in one
window. Persona drift ranks the brains like the behavior test:

| E4B | Persona KL | Same top |
|---|---|---|
| 4-bit QAT | 0.039 | 92.6% |
| brain C | 0.409 | 76.4% |
| mix-21 | 0.482 | 75.1% |
| pipeline mix (generic) | 0.551 | 72.2% |

Even the 4-bit gold answered "30 degrees in Canada" as "could be chilly": the Celsius confusion is the base model's,
not the squeeze's (a fine-tuning target).

## Honest status

- Google's QAT 4-bit is nearly lossless (KL ~0.03 vs 8-bit). 2-bit tracks cost ~15x that. "No quality loss" has to
  come from tensors where 3-bit measures like 4-bit, not from 2-bit.
- Quality-first mode added: the smallest mix whose KL is at or below Google's 4-bit. 31B run in progress on Azure.

## Tools and fixes along the way

- engine: perplexity LG_PPL_SPECIAL=1 (chat markers as control tokens), LG_PPL_ONE_WINDOW=1 (single-window text).
- Benchmark files pinned with checksums (research/big-model/get_bench.sh, bench.sha256).
- Mac storage: the external drive linked at USB 2.0 (480 Mb/s, a Samsung 990 EVO Plus in a 10 Gb/s enclosure): the
  cable. 213 GB of old brains moved there anyway, verified (94 files, same sizes), Mac 40 -> 211 GB free.

## 2026-10-15: 31B quality-target and 16 GB fit

Top-down trim from the official 4-bit (built from donors: 18.10 GB, chat KL 0.0636).

- Quality target (official drift +10%, KL <= 0.070): 1 downgrade, 17.87 GB, KL 0.0685. Only ~1% smaller.
  The QAT 4-bit is the sweet spot; every 2/3-bit track costs more than the budget allows.
- Size target (16 GB, so it fits a 24 GB Mac's GPU limit of 19.07 GB with room for context):

| Round | Downgrades | Size | Chat KL |
|---|---|---|---|
| start | 0 | 18.10 GB | 0.064 |
| 1 | 4 | 16.82 GB | 0.194 |
| 2 | 8 | 16.44 GB | 0.215 |
| 3 | 11 | 15.26 GB | 0.299 |

Same tests, 31B, vs 8-bit:

| 31B | Size | Raw KL | Raw same top | Chat KL | Chat same top |
|---|---|---|---|---|---|
| Google official 4-bit | 17.65 GB | 0.097 | 88.4% | 0.064 | |
| 16 GB fit | 15.26 GB | 0.348 | 75.8% | 0.299 | 79.5% |
| Brain C recipe | 13.86 GB | 0.767 | 62.8% | | |

- The 16 GB fit is less than half brain C's raw drift, and about 3.5x Google's. Not "no quality loss".
- Azure note: the VM was deallocated mid-run (auto-shutdown reset by another tool); the probe results were rebuilt
  from pipeline.log, no re-measurement. The scratch disk is now mounted by UUID so restarts can't move it.

### 16 GB fit on the Mac (M4 Pro 24 GB, Metal, -ngl 99, same session)

Copied from Azure in 8 parallel ssh byte ranges (~7 MB/s vs ~2.5 MB/s single stream), sha256 matched.

| 31B | Talking (tg32) | Reading 1K / 2K / 4K / 8K prompt |
|---|---|---|
| 16 GB fit | 12.6-12.8 tok/s | 105 / 97 / fails at default batch / 95 with -ub 256 |
| Brain C | 14.2-14.3 tok/s | 105 / 103 / 100 / |

- The 16 GB fit fails a 3-4K prompt at the default micro-batch (512): file + context + workspace crosses the 19.07 GB
  GPU limit. With -ub 256 it reads 4K at 99.5 and 8K at 95 tok/s (no real speed cost).
- Swap: 6.5 GB already used by other apps before the tests; 33 MB paged out during the 8K run. Not yet a clean
  "no swap" test.
