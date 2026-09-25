package com.littleguy.robot.core

import android.util.Log
import com.google.ai.edge.litertlm.Backend
import com.google.ai.edge.litertlm.Content
import com.google.ai.edge.litertlm.Contents
import com.google.ai.edge.litertlm.Conversation
import com.google.ai.edge.litertlm.ConversationConfig
import com.google.ai.edge.litertlm.Engine
import com.google.ai.edge.litertlm.EngineConfig
import com.google.ai.edge.litertlm.ExperimentalApi
import com.google.ai.edge.litertlm.ExperimentalFlags
import com.google.ai.edge.litertlm.Message
import com.google.ai.edge.litertlm.MessageCallback
import java.util.concurrent.CountDownLatch
import java.util.concurrent.TimeUnit
import java.util.concurrent.atomic.AtomicLong
import java.util.concurrent.atomic.AtomicReference

class LiteRtLmBrain(
    private val modelPath: String,
    private val cacheDir: String,
    private val backend: Backend = Backend.CPU(),
    private val streamForMeasurement: Boolean = false
) : BrainBackend, AutoCloseable {
    private var engine: Engine? = null
    private var conversation: Conversation? = null

    @OptIn(ExperimentalApi::class)
    @Synchronized
    override fun respond(input: BrainInput): BrainOutput {
        val activeConversation = conversation ?: initialize(input.recentTurns)
        val prompt = renderInput(input)
        val (speech, firstTextMs) = if (streamForMeasurement) {
            streamResponse(activeConversation, prompt, input.onTextChunk)
        } else {
            val response = activeConversation.sendMessage(prompt)
            response.contents.contents
                .filterIsInstance<Content.Text>()
                .joinToString(separator = "") { it.text }
                .trim() to null
        }

        check(speech.isNotEmpty()) { "Gemma returned no spoken text." }
        val inference = runCatching {
            activeConversation.getBenchmarkInfo().let { benchmark ->
                InferenceMetrics(
                    sdkReportedTtftMs = benchmark.timeToFirstTokenInSecond * 1_000,
                    measuredFirstTextMs = firstTextMs,
                    prefillTokens = benchmark.lastPrefillTokenCount,
                    prefillTokensPerSecond = benchmark.lastPrefillTokensPerSecond,
                    decodeTokens = benchmark.lastDecodeTokenCount,
                    decodeTokensPerSecond = benchmark.lastDecodeTokensPerSecond
                )
            }
        }.getOrElse { failure ->
            Log.w("LittleGuyInference", "LiteRT benchmark data unavailable", failure)
            null
        }
        return BrainOutput(speech = speech, inference = inference)
    }

    private fun streamResponse(
        activeConversation: Conversation,
        prompt: String,
        onTextChunk: ((String) -> Unit)?
    ): Pair<String, Double?> {
        val started = System.nanoTime()
        val firstText = AtomicLong(0)
        val error = AtomicReference<Throwable?>()
        val complete = CountDownLatch(1)
        val chunks = StringBuilder()

        activeConversation.sendMessageAsync(prompt, object : MessageCallback {
            override fun onMessage(message: Message) {
                val text = message.contents.contents
                    .filterIsInstance<Content.Text>()
                    .joinToString(separator = "") { it.text }
                if (text.isNotEmpty()) {
                    firstText.compareAndSet(0, System.nanoTime())
                    synchronized(chunks) { chunks.append(text) }
                    onTextChunk?.invoke(text)
                }
            }

            override fun onDone() {
                complete.countDown()
            }

            override fun onError(throwable: Throwable) {
                error.set(throwable)
                complete.countDown()
            }
        })

        if (!complete.await(60, TimeUnit.SECONDS)) {
            activeConversation.cancelProcess()
            throw IllegalStateException("LiteRT streaming did not finish within 60 seconds")
        }
        error.get()?.let { throw it }
        val speech = synchronized(chunks) { chunks.toString().trim() }
        val firstMs = firstText.get().takeIf { it != 0L }
            ?.let { (it - started) / 1_000_000.0 }
        return speech to firstMs
    }

    @OptIn(ExperimentalApi::class)
    private fun initialize(recentTurns: List<String>): Conversation {
        ExperimentalFlags.enableBenchmark = true
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
