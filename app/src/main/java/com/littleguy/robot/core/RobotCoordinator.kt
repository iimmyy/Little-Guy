package com.littleguy.robot.core

class RobotCoordinator(
    private val brain: BrainBackend,
    private val hardware: RobotHardware
) {
    private val turns = ArrayDeque<String>()

    fun receive(text: String): BrainOutput {
        val output = brain.respond(
            BrainInput(
                text = text,
                recentTurns = turns.toList(),
                observation = hardware.observe()
            )
        )

        turns.addLast("Human: $text")
        turns.addLast("Little Guy: ${output.speech}")
        while (turns.size > 12) turns.removeFirst()

        hardware.perform(output.action)
        return output
    }
}
