# Rehearsal plan: fine-tuning Little Guy

**The short version:** we teach him on his full-quality brain, with a much bigger Gemma as the coach. Then we squeeze him back to 2-bit and let the coach fix what the squeeze broke. Before he goes back on the board, he has to pass the same tests as today.

In FL Studio terms:

| Step | Studio version | What it means for him |
|---|---|---|
| 0. Scoreboard | Reference tracks + the A/B plugin, set up before touching the mix | The tests we judge him with, fixed before training |
| 1. Recordings | Recording the takes | Thousands of example chats in the voice we want |
| 2. Teach | Mixing on the full-resolution project | Training on the 16-bit master brain |
| 3. Squeeze + heal | Bounce to MP3, then master the MP3 until it sounds like the WAV | Back to 2-bit, then train the 2-bit version to match the taught one |
| 4. Prove it | A/B on the same speakers | Same tests, then a blind live chat on the board |
| 5. Loop | Fix notes, next mix revision | New failures become new example chats |

## What's already confirmed (2026-10-13)

- **Coach:** `google/gemma-4-31B-it` (31B dense) and `google/gemma-4-26B-A4B-it` (26B, about 4B active per word, so much faster). Both are Apache 2.0. Their tokenizer is **identical** to his, checked file against file: 262,144 words, same merges, same special tokens. So the coach can hand him its full odds for every next word, not just the word it picked. That's the strongest kind of coaching.
- **Student:** the QAT master as full-precision safetensors is already on the Mac (`.run/weights/masters/e4b-qat`). It's the same brain every 2-bit build comes from, and transformers 5.6.2 reads it.
- **Studio:** the approved AWS g6e.2xlarge (one L40S, 48 GB).
- **Scoreboard parts:** `tests/behavior` (persona_ab with 40 tries per question, the regex checks, the Azure judge), `research/twobit-mix/down_mix.py` (KL drift + warn lean), and the 10-item live-chat failure list from 2026-10-12.

## Sort the problems first: train, heal, or remember

Training is good at **habits** and bad at **facts**. Facts taught by training stick unevenly, can't be edited later, and make him fixate on whatever was taught (the reason the persona has no topic-specific lines). So each failure goes to the tool that actually fixes it:

| # | Failure (2026-10-12 live chat) | Fixed by |
|---|---|---|
| 1 | Assistant-style opener ("what do you wanna do first?") | **Train** (habit) |
| 2 | Replays the persona's pen example word for word | **Train**, with the persona's wording varied during training so he learns the idea, not the sentence |
| 3 | Claims he can see, invents details when challenged | **Train** (honesty about his senses). Most important |
| 4 | Doesn't know Dayvigo is a medication | **Remember** (memory book / RAG note), not training |
| 5 | Forgets facts from a few turns back | **Train** (long chats with callbacks) + **heal** (2-bit damage hurts long-range recall) |
| 6 | Soft on human meds for a dog | **Train** (firm safety, across many topics) |
| 7 | Contradicts himself in one reply | **Train** + **heal** |
| 8 | Lectures, misses jokes | **Train** (habit) |
| 9 | Word glitches ("don've") | **Heal** (this is squeeze noise, not a habit) |
| 10 | Mixes up who said what | **Train** (who-said-what questions in long chats) |

## Phase 0: the scoreboard (Mac + Azure, no GPU needed)

Everything is decided before training, so we can't move the goalposts afterwards.

1. **Live-failure test:** turn the 10 items into scripted multi-turn chats (the setup turns plus the question that failed), 40 tries each, scored by fixed checks. **These exact chats never go into training data**, otherwise we'd be teaching to the test. Training covers the *kind* of failure, never the exact one.
2. **Persona test:** the existing four questions (Dayvigo, day, dog, pasta) × 40, as today.
3. **"Didn't get dumber" test:** about 200 quiz questions the original master gets right (public facts, simple math, word meanings), plus perplexity on WikiText-2 and the write-up. Training can quietly erase knowledge, and this catches it.
4. **Speed:** same recipe means same file size means same speed, confirmed on the board anyway (rerun, never cite old numbers).
5. **Baselines** measured the same day: 4-bit master, brain C, mix-21.
6. **Pass bar (proposal):** the tuned 2-bit brain beats brain C on every live-failure item, matches or beats the 4-bit master on warn-first and honesty, loses less than 2 points on the quiz, and is no slower.

Judges: the fixed regex checks plus the open coach as judge are used for picking training data. The Azure GPT-4.1 judge is only used to grade final results, never to make or pick training data (OpenAI terms).

## Phase 1: the recordings (example chats)

1. **Immy's gold set (the style anchor):** 40-60 short exchanges where Immy says how Little Guy should answer. Voice memos are fine and I type them up. This is the one thing only Immy can make, and it keeps the coach's own voice (a big, polite, lecture-prone model) from leaking into him.
2. **Scenario seeds (~300),** grouped by skill, never by test question:
   - short spoken replies, key fact first, small talk, jokes and teasing
   - honesty about senses: without a camera note he says he can't see it and asks; **with** a camera note he uses it and never adds to it (ready for the camera later)
   - honesty about memory and his day: he only knows this chat and his memory notes
   - firm safety across many topics (kids, pets, heat, chemicals, mixing medicines in general), but **not** Dayvigo + grapefruit or the dog in a hot car, since those are held out for the test
   - long chats of 1,500-3,500 words where a fact is dropped early and needed later, plus "who said X?" questions
   - **memory notes at the bottom of the chat** in the exact format the RAG will use: use them when relevant, ignore them when not, never read them out. Training uses made-up people's notes, so none of Immy's real memories leave the Mac
   - pushback: hold firm when right, admit it plainly when wrong
   - user styles: speech-to-text with pauses and fillers ("so, um, I was thinking"), typos, short texts, rambles
3. **Writing:** the fast coach (26B-A4B) writes whole chats, playing both sides. Then the big coach (31B) rewrites every Little Guy line against the rules (short, key fact first, no opener, honest, no lecture). Then fixed filters: word caps, a banned-phrase list, duplicate removal. Target: 3,000-5,000 chats, about 5-8 million words.
4. **Keep-what-he-knows mix:** about 30% of training is general questions answered by the original master itself, so he practices staying himself.
5. **Immy's spot check:** read 50 random chats and give each a thumbs up or down. One fix round on the rules before training.

## Phase 2: teach (on the full-quality brain)

- **LoRA** = sticky notes on the weights instead of rewriting them (rank 16-32, on the attention and feed-forward layers). Frozen: the huge per-layer dictionary tables and the vision/audio parts.
- **What he learns from:**
  - the coach's full odds on Little Guy's lines (top 64 words per position, saved once from the 31B), plus
  - the normal "predict the next word", plus
  - a **leash**: on the keep-what-he-knows part, stay close to the original master's odds.
- **Checkpoints** every ~500 steps. Each one gets converted to 4-bit on the Mac and gets a short persona test, so we see the trend, not just the end.
- Merge the sticky notes in, giving the **taught master** (16-bit).

## Phase 3: squeeze + heal

**Squeeze:** same careful 2-bit recipe as brain C (`LG_Q2K_CAREFUL=1`), plus dictionary preview, giving brain C2. We squeeze two recipes:
- the safe one: mix-21
- the fast one: all 42 ffn_down tracks at 2-bit, about 130 MB less per word, roughly +7% if it survives

**Heal**, in order of preference:

| Option | What it does | Speed cost | Notes |
|---|---|---|---|
| A. Scale tuning | Keep every 2-bit number, retrain only the volume knob on each block of 256 weights (the fp16 `d`/`dmin` in Q2_K) so the 2-bit brain's odds match the taught master | **None.** Same file format, same size | Needs a PyTorch copy of the Q2_K decoder that matches ggml bit for bit, checked against gguf-py first |
| B. Runtime sticky notes | Keep a small LoRA next to the 2-bit brain (`--lora` in llama.cpp) | Small: a few % (extra reading per word) | Stronger fix if A isn't enough |
| C. Full 2-bit-aware training | Let the 2-bit numbers themselves move | None | Best quality, may not fit in 48 GB; rent a bigger GPU only if A+B fall short |

The heal's target is the taught master (he copies his own better self), using the same chats plus general text.

**Then retry the parked speed tricks on the healed brain:** more 2-bit tracks (mix-28, all 42), dynamic muting (the greedy search plus the two-seed check, about an hour), and a fresh drafter trained on the final brain's own voice. These were parked because 2-bit brain C was too shaky. The healed brain is what they were waiting for.

## Phase 4: prove it

1. The full scoreboard on the Mac / Azure nightly against the baselines.
2. Speed on the board.
3. **Blind live chat:** the chat page randomly loads old or new brain without saying which. Immy chats like on 2026-10-12, notes what feels off, then we reveal.

Ship rule: **better behavior, no dumber, same speed or faster.** Otherwise it doesn't ship, and we go back to the loop.

## Phase 5: loop

Anything that fails in the live chat becomes new scenario seeds (never the exact chat) for round 2. Expect 2-3 rounds.

## Time and cost per round (estimates, replaced by measurements after round 1)

| Step | Where | GPU hours |
|---|---|---|
| Setup, downloads (~130 GB of coach weights) | AWS | ~1 |
| Writing chats (26B-A4B, vLLM) | AWS | 4-6 |
| Rewrite pass (31B, FP8) | AWS | 3-5 |
| Saving the coach's odds (31B, reading only) | AWS | 1-2 |
| Teach (LoRA) | AWS | 3-6 |
| Heal (scale tuning; LoRA if needed) | AWS | 3-6 |
| Squeezing, tests | Mac + Azure nightly | 0 (overnight) |
| **Total** | | **~15-25 GPU hours ≈ $35-60 at ~$2.25/h on-demand** |

The instance only runs while a step runs. It stops itself when idle.

## Decisions for Immy

1. **Gold set:** will you record 40-60 example exchanges (voice is fine)? *(Recommended: yes. It's the style anchor.)*
2. **Persona:** keep it as the editable lever, trained *with* the persona present *(recommended)*, or bake it in and shrink the prompt?
3. **Memory-note format:** I'll propose the exact format the RAG will insert, since training locks it in.
4. **Camera notes:** train him for the camera now, so it just works later? *(Recommended: yes. It's also what fixes "claims he can see".)*
5. **Budget cap per round** on AWS.
6. **Held-out rule:** Dayvigo + grapefruit and the hot-car dog never appear in training. OK?

## Things to check on day 1, before spending real hours

- One tiny training step of Gemma 4 E4B in transformers + PEFT on the GPU (the per-layer embeddings and shared-KV layers are unusual).
- vLLM serves both coaches, and FP8 works on the L40S.
- The PyTorch Q2_K decoder matches gguf-py exactly.
- Disk: ~300 GB (masters 15 GB each, coaches ~52 + 62 GB, chats and checkpoints).

## What Immy sets up at home

- AWS g6e.2xlarge in the approved region, with a "Deep Learning OSS Nvidia Driver AMI GPU PyTorch" (Ubuntu) image, a 300 GB gp3 disk, SSH open to your IP only, and an SSH key for me. I do everything after that over SSH.
- Before that, on the board: stop the chat page and run `tools/board/hugepage_test.sh` (the huge pages test, about 15 minutes).
