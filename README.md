# Little Guy

I'm building a small Q6A-based robot companion that can perceive its surroundings, remember interactions, and express itself through movement and sound.

## Current milestone

- I have an Android 15 conversation lab running in an ARM64 emulator.
- I can replace the brain and robot hardware backends independently.
- The brain proposes structured actions instead of controlling hardware directly.
- A safety gate blocks actions outside the approved movement policy.
- Conversation memory survives a full app restart and stays outside the model.
- Each interaction reports brain and total response time.
- Camera, sound-direction, distance, and motion data are simulated for now.

Gemma is not integrated yet. I'm deliberately using a small deterministic development brain so I can prove the memory, safety, timing, and hardware boundaries without pretending the emulator reproduces the Q6A's Qualcomm NPU.

I can currently try `hello`, `look`, `dance`, `turn around`, `what did I say?`, `what can you see?`, or `how close is it?` in the lab. `turn around` intentionally requests an unsafe 180-degree movement so I can verify that the safety gate blocks it.

## Hardware deployment path

1. I will bring up the official Dragon Q6A Android image on the 256 GB UFS module first.
2. I will measure real startup time, time to first response, sustained generation speed, memory use, and thermals before locking the OS or model.
3. I will replace `DevelopmentBrain` with the selected LiteRT/QNN model-runtime adapter.
4. I will replace `SimulatedHardware` one device at a time with USB or serial adapters for the Alpha controller, microphone array, LiDAR, and cameras.
5. I will validate every model and hardware path on the physical Q6A.

I am not doing heavy model training on my Mac or in this Android project. I can prepare behaviour data here, then run any later LoRA fine-tuning on suitable cloud hardware before exporting and quantizing the result for the Q6A.

I am keeping the codebase suitable for a future public release. Model weights, Qualcomm vendor payloads, secrets, signing material, private drafts, and local build output stay out of Git.
