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

## Next physical-board gate

On the Q6A I will record:

- cold and warm initialization time;
- time to first token and time to first complete response;
- prefill and decode speed;
- peak memory use;
- sustained thermals and throttling;
- whether GPU is sufficient or a QNN/NPU-specific export is worth pursuing.

Until that run succeeds, I am treating Gemma inference on the Q6A as unverified.
