# Little Guy

Android control environment for the Q6A-based robot.

## Current milestone

- Emulator-ready conversation lab
- Replaceable brain backend
- Replaceable robot hardware backend
- Simulated camera, sound-direction, distance, and motion data

The development backend is deliberately fake. It lets the interaction loop and robot behaviour evolve without pretending the Android emulator can reproduce the Q6A's Qualcomm NPU.

Try `hello`, `look`, `dance`, `what can you see?`, or `how close is it?` in the lab.

## Hardware deployment path

1. Flash the official Dragon Q6A Android UFS image to the 256 GB module.
2. Replace `DevelopmentBrain` with the LiteRT/QNN inference backend.
3. Replace `SimulatedHardware` with USB/serial adapters for the Alpha controller, microphone array, LiDAR, and cameras.
4. Validate model latency and each hardware path on the Q6A.

Heavy model training does not run in this project. Behaviour data can be prepared here; any later LoRA fine-tuning should run on suitable cloud hardware, then be exported and quantized for the Q6A.
22