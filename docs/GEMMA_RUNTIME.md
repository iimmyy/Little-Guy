# Gemma runtime validation (WIP)

I am using Google's official generic LiteRT-LM build of Gemma 4 E2B:

- Source: [litert-community/gemma-4-E2B-it-litert-lm](https://huggingface.co/litert-community/gemma-4-E2B-it-litert-lm)
- File: `gemma-4-E2B-it.litertlm`
- Size: `2,588,147,712` bytes
- SHA-256: `181938105e0eefd105961417e8da75903eacda102c4fce9ce90f50b97139a63c`

The model is intentionally excluded from Git. The Android app expects it at:

```text
/sdcard/Android/data/com.littleguy.robot/files/models/gemma-4-E2B-it.litertlm
```

## What I have verified

- The Kotlin adapter builds against LiteRT-LM `0.17.1` with Kotlin `2.4.20` and JDK 17.
- The app detects the model, creates the engine away from the UI thread, restores external conversation memory, and asks Gemma for spoken text.
- The GPU backend registered successfully in the Android 15 ARM64 emulator.
- LiteRT loaded all four Gemma graphs and delegated every node to its WebGPU path.
- Unit tests and the debug APK build pass.

## Emulator boundary

The emulator does not have a real Adreno GPU. LiteRT fell back to WebGPU on Mesa `llvmpipe`, which is a software renderer. It created a 741 MB weight cache and an 11 MB program cache, but a first short response still had not completed after more than four minutes. That is evidence that the model and GPU graphs load; it is not a useful performance result.

I also tried the LiteRT CPU backend only to isolate the emulator. Its native library exited with `SIGILL` during `nativeCreateEngine`, so I did not keep that fallback in the app. The deployable path remains GPU, as intended for the Q6A.

## First physical Q6A run

I copied the exact model to the Q6A, verified its SHA-256 there, and sent two short prompts through the Android app while monitoring thermal zones from the Mac. The first `Hi` produced `Hello. I see you.` in 50.696 seconds end-to-end, including cold engine initialization. A second `Hi again` produced the same short answer in 1.787 seconds end-to-end with the engine warm. These are full-response times, **not** time-to-first-token measurements.

The Q6A log reports full LiteRT OpenCL delegation for the prefill and verify subgraphs. The generic LiteRT NPU accelerator did not register, and the optional OpenCL top-k sampler library was absent, so it fell back to its statically linked sampler. Those warnings did not prevent the two responses. The process stayed alive as a foreground service after I returned to Home.

With a small external fan blowing on the open board, the logged maxima during this first run were 34.9 C across CPU thermal zones, 37.2 C across GPU zones, and 29.9 C at the named skin zone; Android thermal status was 0. The app's reported total PSS after loading was about 1.90 GB. This short two-prompt test is not a sustained-load or enclosed-robot thermal result, and I still lack a direct UFS temperature reading.

I still need varied prompt lengths, screen-off behavior, and sustained thermal testing before choosing a final model or cooling design.

## Instrumentation follow-up (WIP)

I enabled LiteRT-LM's experimental benchmark counters in the debug build. A warm `Hi again` completed in 1.997 seconds; LiteRT reported 31 prefill tokens at 41.1 tokens/s and 8 decode tokens at 6.7 tokens/s. Its reported TTFT stayed exactly 1,917 ms across both the preceding cold run and that warm run, so I am not accepting it as a per-turn warm TTFT measurement.

I then timestamped the first nonempty text callback from `sendMessageAsync` on the device clock. The cold run produced its first text 4.352 seconds after generation began and finished in 49.103 seconds including engine initialization. Three subsequent short warm prompts produced first text in 1.121, 1.109, and 1.128 seconds; their complete responses took 1.873, 1.952, and 1.896 seconds. These are first-*text* timings, not verified token timestamps or time to first spoken audio. LiteRT's own TTFT counter remained near 1.91 seconds and is still labelled unverified. The debug-only stream path has a 60-second completion ceiling so a stalled callback cannot hang this experiment indefinitely.

The short warm prompts used roughly 30 prefill tokens at 44 tokens/s and seven decode tokens at 5.6–6.0 tokens/s. That makes the initial text delay the first latency target. KV-cache policy and conversation compression may matter as history grows, but these tiny prompts do not establish a KV-memory bottleneck. I will benchmark longer conversations before changing cache size or quantization.

## Q6A speed probes (WIP)

After a fresh board boot, three identical warm `Hi` turns with the unchanged simulated observation reached first text in 1.025, 1.078, and 1.096 seconds. I then packaged Google's `v0.17.1` ARM64 `libLiteRtTopKOpenClSampler.so` in a temporary debug build because the original log reported that optional library missing. The three warm turns in that build reached first text in 1.077, 1.102, and 1.118 seconds. This does not demonstrate a first-text improvement; I reinstalled the known-good APK without the extra native library. I did not retain the binary in Git.

I also briefly enabled Android's fixed-performance mode while the external fan and thermal monitor ran. Three warm first-text timings were 1.069, 1.055, and 1.089 seconds, again not a meaningful improvement over the normal-clock baseline. I turned fixed-performance mode back off. That mode is a benchmark control, not a permanent robot setting.

The Q6A's Gemma graph runs through Adreno/OpenCL, not Hexagon. Qualcomm identifies this board's QCS6490 NPU as a 12-TOPS component, and Radxa documents its Hexagon target as v68. The public LiteRT-LM Gemma 4 E2B package currently lists Qualcomm NPU files for QCS8275 and SM8750, not QCS6490. I cannot claim NPU acceleration or reuse those other-chip binaries. The next NPU gate is to validate the v68 runtime with a known-good QCS6490 model, then obtain or compile a compatible Gemma artifact and measure actual delegation, output quality, first-text latency, power, and heat. Radxa's published quick validation is on Ubuntu, so that exact procedure is not yet an Android test.

**Update (2026-09-28):** Yeah so… I ended up just moving the board to Ubuntu instead of fighting Android for the NPU. The v68 check passed there with Radxa's Llama 3.2 1B bundle, it's all in [Q6A_LINUX_BOOT.md](Q6A_LINUX_BOOT.md). Everything above this is my Android GPU work, and it still stands as my GPU baseline.

The present 30-token `Hi` benchmark is diagnostic, not a stand-in for Little Guy operating with cameras, microphones, place state, and conversation history. I need a fixed representative and a deliberately heavy sensor snapshot before accepting a speed optimization as useful for the robot.

## Longer conversational prompt (WIP)

After a fresh Q6A boot, I prompted the headless service: “I'm carrying you into a coffee shop. You cannot see or hear it yet. What would you say to me in one short sentence?” The first attempt did not reach the model: `q6a-ask.py` had passed the apostrophe through ADB shell without remote-shell quoting, producing `no closing quote`. I fixed the shell argument construction and added a regression test.

The successful cold run answered “Hello. I am ready.” First streamed text arrived at 6.449 seconds, and the complete interaction took 54.238 seconds (54.226 seconds inside the brain). LiteRT reported 261 prefill tokens at 74.4 tok/s and seven decode tokens at 2.0 tok/s; its 4.021-second TTFT counter remains unverified against the callback clock. Saved conversation history contributed to the larger prefill. This was a single cold run, not a new warm-performance baseline. With the external fan on and the board open, a post-run sample showed 38.7 C at `cpu-0-0-usr`, 34.5 C at `gpuss-0-usr`, and 35.2 C at the named skin zone. This is not a sustained thermal test.

With the engine still warm, a 45-token prefill reached first text in 1.086 seconds and completed in 1.743 seconds. A longer, explicitly hypothetical coffee-shop prompt used 98 prefill tokens, reached first text in 1.432 seconds, and completed in 1.570 seconds. The latter reply was only “Okay,” so latency alone is not an interaction-quality result. LiteRT's reported TTFT remained frozen at 4.021 seconds across both warm turns, further evidence that this SDK counter is not per-turn timing. A final post-run sample read 39.5 C CPU, 35.2 C GPU, and 36.1 C skin; no enclosure or sustained-load claim follows from these brief runs.

## Debug terminal streaming (WIP)

I now route LiteRT-LM text callbacks through the coordinator to the debug prompt receiver and stream them to the terminal as they arrive. The production service's action decision and safety gate remain unchanged. I base64-encode each log chunk to preserve whitespace and avoid interpreting generated text as a log protocol field. The final response is still committed to memory and the dashboard only after generation completes.

I rebuilt and installed the debug APK on the Q6A. Its first request after an app-process restart took 18.391 seconds end-to-end and 4.943 seconds from generation start to first text. This is not a fresh-board-boot comparison: Android and LiteRT caches could still be warm. A subsequent warm 50-prefill-token prompt reached first text in 1.244 seconds and completed in 3.594 seconds. The Q6A log records the first `CHUNK64` at 00:08:57.743 and the final answer at 00:09:00.095, proving that the terminal path can show text about 2.35 seconds before completion for that response. The SDK TTFT counter again stayed fixed across turns. With the external fan on the open board, the named GPU zone briefly reached 46.1 C during the app-restart test; no sustained/enclosed thermal conclusion follows.

References: [LiteRT-LM Gemma package](https://huggingface.co/litert-community/gemma-4-E2B-it-litert-lm/tree/main), [Radxa Q6A NPU guide](https://docs.radxa.com/en/dragon/q6a/app-dev/npu-dev/qairt-sdk), [Radxa's Ubuntu NPU validation](https://docs.radxa.com/en/dragon/q6a/app-dev/npu-dev/quick-example), [Android fixed-performance mode](https://developer.android.com/games/optimize/adpf/fixed-performance-mode).
