# Little Guy model selection (WIP)

I am treating [Gemma 4 E2B](https://huggingface.co/litert-community/gemma-4-E2B-it-litert-lm) as the working Q6A baseline, not a final choice. It runs on the board through LiteRT-LM and Adreno/OpenCL. My initial debug measurements, including their limits, are in [Gemma runtime validation](GEMMA_RUNTIME.md). I have not verified NPU execution, image input, or audio input on my Q6A.

## Candidates to test

| Candidate | Why I would test it | What remains unproven here |
| --- | --- | --- |
| [Gemma 4 E2B](https://ai.google.dev/gemma/docs/core/model_card_4) | Google lists text, image, and audio input; I already have a working Q6A text baseline. | End-to-end multimodal support in my Android package, sustained power/heat, and QCS6490 NPU compilation. |
| [Gemma 4 E4B](https://huggingface.co/litert-community/gemma-4-E4B-it-litert-lm) | Same model family and LiteRT-LM runtime, with a larger on-device model that may improve difficult interactions. | Whether the quality gain is worth its larger memory footprint and latency on my 12 GB board. |
| [Qwen3.5-2B](https://huggingface.co/Qwen/Qwen3.5-2B) | A different small instruction-tuned vision-language model, useful as an independent quality comparison. | An equivalent Android runtime, audio path, QCS6490 NPU compilation, and fair on-device speed comparison. I will not assume its model-card demos are a deployable Q6A package. |

I will not choose from parameter count or another device's benchmark. I need the same small test set for each candidate: short conversation, a long-history follow-up, a changing-place scenario, uncertainty about absent sensors, a scene description, an audio input, and a request for an unsafe physical action. I will save the exact prompts, outputs, model hashes, runtime/backend, and pass/fail notes. A model that invents observations or bypasses the separate motion safety gate fails even if its text benchmark is fast.

## E4B test setup (WIP)

I have prepared my Android app to select E2B or E4B without changing the conversation prompt, GPU backend, safety gate, or APK build. The selection is the one-line file `models/selected-model.txt` in the app's external files directory. With no selection file, the app still defaults to E2B. I use the tracked `config/model-e2b.txt` and `config/model-e4b.txt` files to avoid mistyping the choice. The debug terminal prints the active model ID and refuses to treat the development brain as a model benchmark. If I explicitly select a missing model or enter an unknown ID, it reports an error instead of silently switching models.

My E4B candidate is the generic, multimodal-capable [LiteRT-LM release](https://huggingface.co/litert-community/gemma-4-E4B-it-litert-lm/tree/main), not its smaller GPU-only variant. I pinned Hugging Face revision `2eee7ac325f20eb8c9ac1d0e972f7c84663062da`. The file `gemma-4-E4B-it.litertlm` is 3,659,530,240 bytes; I verified its local SHA-256 as `0b2a8980ce155fd97673d8e820b4d29d9c7d99b8fa6806f425d969b145bd52e0`. I keep this multi-gigabyte file under `.run/models/` locally and out of Git. Downloading and building the app do not yet prove the file loads or generates on my Q6A.

When the Q6A is powered, cooled, and connected over ADB, I will verify its free space, copy the file to `/sdcard/Android/data/com.littleguy.robot/files/models/gemma-4-E4B-it.litertlm`, compare its on-device SHA-256 to the expected hash, push `config/model-e4b.txt` to `/sdcard/Android/data/com.littleguy.robot/files/models/selected-model.txt`, then restart the service. To return to the known E2B baseline, I will push `config/model-e2b.txt` to the same selector path and restart the service. Model selection is read when the service starts, not in the middle of a response.

This first E4B test is still Adreno/OpenCL. It is not NPU conversion, retraining, or a final model decision. I will only compare E2B and E4B after recording the actual Q6A backend, output, first visible text, full response, memory, and thermals under equivalent conditions.

For every model that actually runs on my Q6A, I will measure first visible text, complete-response time, prefill/decode throughput, memory, temperature, and power under the same fan and prompt conditions. I will distinguish a fresh boot, an app-process restart, and a warm conversation. Model support for image or audio on paper is only a candidate capability until I send those modalities through the exact deployed runtime.

## NPU compilation gate

The QCS6490 targets Hexagon **v68** with SoC ID **35**. [Radxa's QAIRT guide](https://docs.radxa.com/en/dragon/q6a/app-dev/npu-dev/qairt-sdk) says a context binary is chip-specific, and its [conversion example](https://docs.radxa.com/en/dragon/q6a/app-dev/npu-dev/qairt-usage) demonstrates quantization and context-binary generation on an x86 Linux host for a smaller vision model. Those instructions do not establish that I can compile Gemma 4 or Qwen3.5 on this Mac, nor that a compiled artifact will run on this Android image.

Before attempting a full LLM conversion, I will verify the NPU runtime with a known-good QCS6490 example, confirm the Android libraries/toolchain and model-export path, then try one chosen model. This is deployment compilation, not retraining. I will keep the working GPU build available as a fallback and require actual NPU delegation, correct output, and measured latency/power gains before calling the NPU route successful.
