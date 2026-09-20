package com.littleguy.robot.core

class DevelopmentBrain : BrainBackend {
    override fun respond(input: BrainInput): BrainOutput {
        val normalized = input.text.trim().lowercase()

        return when {
            normalized.contains("hello") || normalized.contains("hey") -> BrainOutput(
                speech = "Yo. I'm awake—mostly.",
                action = RobotAction.Gesture("wave")
            )

            normalized.contains("look") -> BrainOutput(
                speech = "Looking over there. Probably.",
                action = RobotAction.TurnToward(30)
            )

            normalized.contains("dance") -> BrainOutput(
                speech = "This is either dancing or a calibration failure.",
                action = RobotAction.Gesture("dance")
            )

            normalized.contains("turn around") -> BrainOutput(
                speech = "Requesting a full turn.",
                action = RobotAction.TurnToward(180)
            )

            normalized.contains("what did i say") -> {
                val previousHumanTurn = input.recentTurns
                    .lastOrNull { it.startsWith(HUMAN_PREFIX) }
                    ?.removePrefix(HUMAN_PREFIX)

                BrainOutput(
                    speech = previousHumanTurn?.let { "You previously said: $it" }
                        ?: "I don't have an earlier message from you."
                )
            }

            normalized.contains("see") -> BrainOutput(
                speech = if (input.observation.faceDetected) {
                    "I can see somebody in front of me. Simulated somebody, for now."
                } else {
                    "I don't see anyone right now."
                }
            )

            normalized.contains("close") || normalized.contains("distance") -> BrainOutput(
                speech = input.observation.distanceCm?.let { distance ->
                    "The nearest thing is about $distance centimetres away."
                } ?: "My distance sensor isn't available."
            )

            else -> BrainOutput(
                speech = "I heard: ${input.text.trim()}. My real brain isn't plugged in yet."
            )
        }
    }

    private companion object {
        const val HUMAN_PREFIX = "Human: "
    }
}
