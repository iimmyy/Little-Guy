package com.littleguy.robot.core

import com.google.ai.edge.litertlm.Backend
import com.google.ai.edge.litertlm.Content
import com.google.ai.edge.litertlm.Contents
import com.google.ai.edge.litertlm.Conversation
import com.google.ai.edge.litertlm.ConversationConfig
import com.google.ai.edge.litertlm.Engine
import com.google.ai.edge.litertlm.EngineConfig
import com.google.ai.edge.litertlm.Message

class LiteRtLmBrain(
    private val modelPath: String,
    private val cacheDir: String,
    private val backend: Backend = Backend.CPU()
) : BrainBackend, AutoCloseable {
    private var engine: Engine? = null
    private var conversation: Conversation? = null

    @Synchronized
    override fun respond(input: BrainInput): BrainOutput {
        val activeConversation = conversation ?: initialize(input.recentTurns)
        val response = activeConversation.sendMessage(renderInput(input))
        val speech = response.contents.contents
            .filterIsInstance<Content.Text>()
            .joinToString(separator = "") { it.text }
            .trim()

        check(speech.isNotEmpty()) { "Gemma returned no spoken text." }
        return BrainOutput(speech = speech)
    }

    private fun initialize(recentTurns: List<String>): Conversation {
        val newEngine = Engine(
            EngineConfig(
                modelPath = modelPath,
                backend = backend,
                cacheDir = cacheDir
            )
        )

        try {
            newEngine.initialize()
            val newConversation = newEngine.createConversation(
                ConversationConfig(
                    systemInstruction = Contents.of(SYSTEM_INSTRUCTION),
                    initialMessages = recentTurns.mapNotNull(::toMessage),
                    maxOutputToken = 192
                )
            )
            engine = newEngine
            conversation = newConversation
            return newConversation
        } catch (failure: Throwable) {
            runCatching { newEngine.close() }
            throw failure
        }
    }

    private fun renderInput(input: BrainInput): String = buildString {
        appendLine("Current robot observation:")
        appendLine("- face_detected: ${input.observation.faceDetected}")
        appendLine("- distance_cm: ${input.observation.distanceCm ?: "unknown"}")
        appendLine()
        append(input.text.trim())
    }

    private fun toMessage(turn: String): Message? = when {
        turn.startsWith(HUMAN_PREFIX) -> Message.user(turn.removePrefix(HUMAN_PREFIX))
        turn.startsWith(ROBOT_PREFIX) -> Message.model(turn.removePrefix(ROBOT_PREFIX))
        else -> null
    }

    @Synchronized
    override fun close() {
        conversation?.close()
        conversation = null
        engine?.close()
        engine = null
    }

    private companion object {
        const val HUMAN_PREFIX = "Human: "
        const val ROBOT_PREFIX = "Little Guy: "
        const val SYSTEM_INSTRUCTION =
            "You are Little Guy, a tiny companion robot. Reply with only the short, natural words " +
                "you want spoken aloud. Be warm, curious, concise, and honest about the supplied " +
                "sensor state. Do not claim that you moved or sensed something unless the input says so."
    }
}
