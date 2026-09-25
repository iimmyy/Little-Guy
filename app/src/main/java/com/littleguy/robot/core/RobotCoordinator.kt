package com.littleguy.robot.core

class RobotCoordinator(
    private val brain: BrainBackend,
    private val hardware: RobotHardware,
    private val memory: ConversationMemory = InMemoryConversationMemory(),
    private val safetyGate: ActionSafetyGate = ActionSafetyGate(),
    private val nanoTime: () -> Long = System::nanoTime
) {
    fun receive(text: String, onTextChunk: ((String) -> Unit)? = null): InteractionResult {
        val started = nanoTime()
        val brainStarted = nanoTime()
        val output = brain.respond(
            BrainInput(
                text = text,
                recentTurns = memory.recentTurns(12),
                observation = hardware.observe(),
                onTextChunk = onTextChunk
            )
        )
        val brainFinished = nanoTime()

        memory.append(
            listOf(
                "Human: $text",
                "Little Guy: ${output.speech}"
            )
        )

        val decision = safetyGate.evaluate(output.action)
        if (decision is ActionDecision.Allowed) {
            hardware.perform(decision.action)
        }

        val finished = nanoTime()
        return InteractionResult(
            output = output,
            actionDecision = decision,
            timing = InteractionTiming(
                brainMs = (brainFinished - brainStarted) / NANOS_PER_MILLISECOND,
                totalMs = (finished - started) / NANOS_PER_MILLISECOND
            )
        )
    }

    private companion object {
        const val NANOS_PER_MILLISECOND = 1_000_000.0
    }
}

data class InteractionResult(
    val output: BrainOutput,
    val actionDecision: ActionDecision,
    val timing: InteractionTiming
)

data class InteractionTiming(
    val brainMs: Double,
    val totalMs: Double
)
