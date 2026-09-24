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

I still need token-stream timestamps, prefill/decode throughput, repeated warm trials, screen-off behavior, and sustained thermal testing before choosing a final model or cooling design.
