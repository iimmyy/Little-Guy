# The Little Guy 🫳🏼

As most know, coders are introverts… so instead of making real friends I decided to build one 😁. A Q6A-based robot companion that can perceive its surroundings, remember interactions, and express itself through movement and sound.

My reasoning for the q6a will be discussed in a different document, as my part list grows I'll add them here though.

## Current milestone

- I have an Android 15 conversation lab running in an ARM64 emulator.
- I can replace the brain and robot hardware backends independently.
- The brain proposes structured actions instead of controlling hardware directly.
- A safety gate blocks actions outside the approved movement policy.
- Conversation memory survives a full app restart and stays outside the model.
- Each interaction reports brain and total response time.
- Camera, sound-direction, distance, and motion data are simulated for now.
- I have connected the official Gemma 4 E2B LiteRT-LM model to the app's real brain interface and prepared an explicit E4B test selector.
- I moved the robot runtime into a foreground Android service, leaving the activity as an optional dashboard.
- I added a pure Alpha 1S frame codec, stream decoder, and fake byte transport without exposing motor commands.
- I booted the physical Q6A without HDMI, verified that its foreground service starts on boot, and got the first two Gemma responses on its GPU.

The app now defaults to Gemma E2B when the model is present in its private Android storage. Without a selection file or model it uses my deterministic development brain for emulator work; the debug terminal refuses to benchmark that mock brain. An explicit E4B selection must have the matching model file or the app reports an error. Model initialization and generation run away from the UI thread, and the deployable path uses the Android GPU backend. I have measured first streamed text on the Q6A, but I have not yet measured sustained generation or tested E4B on the board. I recorded the exact boundary in [my Gemma runtime notes](docs/GEMMA_RUNTIME.md) and the [model-selection notes](docs/MODEL_SELECTION.md).

The dashboard no longer owns Gemma, memory, the coordinator, or the hardware backend. `LittleGuyService` owns them and publishes a bounded runtime transcript to any connected dashboard. On the physical Q6A, it now starts after boot without HDMI and stays alive when I return Home. I recorded the architecture and remaining hardware-lifecycle boundary in [my headless runtime notes](docs/HEADLESS_RUNTIME.md).

The first Alpha integration is intentionally hardware-free. The codec implements the published `FB BF ... ED` framing, length, and checksum rules; the stream decoder tolerates split reads and leading noise; and the fake transport lets me test writes and deterministic replies without connecting a robot. The public request API currently exposes only handshake, heartbeat, state, version, battery, and angle reads. I recorded the evidence and unresolved hardware assumptions in [my Alpha protocol notes](docs/ALPHA_PROTOCOL.md).

For the physical Q6A, I prepared a non-destructive macOS EDL detector and an ADB baseline collector. They keep cable detection and evidence capture separate from storage writes, and they refuse to guess when multiple Android devices are attached. The flashing boundary and first-boot gates are in [my Q6A bring-up notes](docs/Q6A_BRINGUP.md).

I can currently try `hello`, `look`, `dance`, `turn around`, `what did I say?`, `what can you see?`, or `how close is it?` in the lab. `turn around` intentionally requests an unsafe 180-degree movement so I can verify that the safety gate blocks it.

## Hardware deployment path

1. I will bring up the official Dragon Q6A Android image on the 256 GB UFS module first.
2. I will measure real startup time, time to first response, sustained generation speed, memory use, and thermals before locking the OS or model.
3. I will run the current LiteRT-LM adapter on the Q6A GPU and record its first real response.
4. I will replace `SimulatedHardware` one device at a time with USB or serial adapters for the Alpha controller, microphone array, LiDAR, and cameras.
5. I will validate every model and hardware path on the physical Q6A.

I am not doing heavy model training on my Mac or in this Android project. I will prepare behaviour data here, then run any later LoRA fine-tuning on suitable cloud hardware (deciding to maybe invest in a personal rig depending the time it takes ofc) before exporting and quantizing the result for the Q6A.

I am keeping the codebase suitable for a future public release. Model weights, Qualcomm vendor payloads, secrets (hehehehehe), signing material, private drafts, and local build output stay out of Git. At least for now….
