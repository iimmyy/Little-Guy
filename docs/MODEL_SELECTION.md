# Little Guy model selection (WIP)

I am treating [Gemma 4 E2B](https://huggingface.co/litert-community/gemma-4-E2B-it-litert-lm) as the working Q6A baseline, not a final choice. It runs on the board through LiteRT-LM and Adreno/OpenCL. My initial debug measurements, including their limits, are in [Gemma runtime validation](GEMMA_RUNTIME.md). I have not verified NPU execution, image input, or audio input on my Q6A.

## Candidates to test

| Candidate | Why I would test it | What remains unproven here |
| --- | --- | --- |
| [Gemma 4 E2B](https://ai.google.dev/gemma/docs/core/model_card_4) | Google lists text, image, and audio input; I already have a working Q6A text baseline. | End-to-end multimodal support in my Android package, sustained power/heat, and QCS6490 NPU compilation. |
| [Gemma 4 E4B](https://huggingface.co/litert-community/gemma-4-E4B-it-litert-lm) | Same model family and LiteRT-LM runtime, with a larger on-device model that may improve difficult interactions. | Whether the quality gain is worth its larger memory footprint and latency on my 12 GB board. |
| [Qwen3.5-2B](https://huggingface.co/Qwen/Qwen3.5-2B) | A different small instruction-tuned vision-language model, useful as an independent quality comparison. | An equivalent Android runtime, audio path, QCS6490 NPU compilation, and fair on-device speed comparison. I will not assume its model-card demos are a deployable Q6A package. |

I will not choose from parameter count or another device's benchmark. I need the same small test set for each candidate: short conversation, a long-history follow-up, a changing-place scenario, uncertainty about absent sensors, a scene description, an audio input, and a request for an unsafe physical action. I will save the exact prompts, outputs, model hashes, runtime/backend, and pass/fail notes. A model that invents observations or bypasses the separate motion safety gate fails even if its text benchmark is fast.

For every model that actually runs on my Q6A, I will measure first visible text, complete-response time, prefill/decode throughput, memory, temperature, and power under the same fan and prompt conditions. I will distinguish a fresh boot, an app-process restart, and a warm conversation. Model support for image or audio on paper is only a candidate capability until I send those modalities through the exact deployed runtime.

## NPU compilation gate

The QCS6490 targets Hexagon **v68** with SoC ID **35**. [Radxa's QAIRT guide](https://docs.radxa.com/en/dragon/q6a/app-dev/npu-dev/qairt-sdk) says a context binary is chip-specific, and its [conversion example](https://docs.radxa.com/en/dragon/q6a/app-dev/npu-dev/qairt-usage) demonstrates quantization and context-binary generation on an x86 Linux host for a smaller vision model. Those instructions do not establish that I can compile Gemma 4 or Qwen3.5 on this Mac, nor that a compiled artifact will run on this Android image.

Before attempting a full LLM conversion, I will verify the NPU runtime with a known-good QCS6490 example, confirm the Android libraries/toolchain and model-export path, then try one chosen model. This is deployment compilation, not retraining. I will keep the working GPU build available as a fallback and require actual NPU delegation, correct output, and measured latency/power gains before calling the NPU route successful.
