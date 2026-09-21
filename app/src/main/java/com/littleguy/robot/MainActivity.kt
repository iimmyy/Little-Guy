package com.littleguy.robot

import android.app.Activity
import android.content.res.ColorStateList
import android.graphics.Color
import android.os.Bundle
import android.text.method.ScrollingMovementMethod
import android.view.Gravity
import android.view.ViewGroup
import android.view.WindowInsets
import android.widget.Button
import android.widget.EditText
import android.widget.LinearLayout
import android.widget.TextView
import com.littleguy.robot.core.ActionDecision
import com.littleguy.robot.core.BrainBackend
import com.littleguy.robot.core.DevelopmentBrain
import com.littleguy.robot.core.LiteRtLmBrain
import com.littleguy.robot.core.RobotCoordinator
import com.littleguy.robot.core.SimulatedHardware
import com.littleguy.robot.platform.SharedPreferencesConversationMemory
import com.google.ai.edge.litertlm.Backend
import java.io.File
import java.util.Locale
import java.util.concurrent.Executors

class MainActivity : Activity() {
    private lateinit var transcript: TextView
    private lateinit var coordinator: RobotCoordinator
    private val brainExecutor = Executors.newSingleThreadExecutor()
    private var brainResource: AutoCloseable? = null

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        val modelFile = File(getExternalFilesDir(null) ?: filesDir, MODEL_RELATIVE_PATH)
        val brain: BrainBackend
        val brainStatus: String
        if (modelFile.isFile) {
            val gemmaBrain = LiteRtLmBrain(
                modelPath = modelFile.absolutePath,
                cacheDir = cacheDir.absolutePath,
                backend = Backend.GPU()
            )
            brain = gemmaBrain
            brainResource = gemmaBrain
            brainStatus = "Gemma model found; it will load on the first message using the device GPU."
        } else {
            brain = DevelopmentBrain()
            brainStatus = "Development brain online. Gemma model not found at:\n${modelFile.absolutePath}"
        }

        transcript = TextView(this).apply {
            setTextColor(Color.rgb(225, 229, 238))
            textSize = 16f
            movementMethod = ScrollingMovementMethod()
            setPadding(32, 24, 32, 24)
            text = "SYSTEM  $brainStatus\n\n"
        }

        coordinator = RobotCoordinator(
            brain = brain,
            hardware = SimulatedHardware(::appendLine),
            memory = SharedPreferencesConversationMemory(
                getSharedPreferences("little_guy_memory", MODE_PRIVATE)
            )
        )

        val input = EditText(this).apply {
            hint = "Say something to Little Guy"
            setSingleLine(true)
            setTextColor(Color.WHITE)
            setHintTextColor(Color.rgb(165, 170, 180))
            backgroundTintList = ColorStateList.valueOf(Color.rgb(92, 107, 192))
        }

        val send = Button(this).apply {
            text = "Send"
            setTextColor(Color.WHITE)
            backgroundTintList = ColorStateList.valueOf(Color.rgb(73, 88, 180))
            setOnClickListener {
                val message = input.text.toString().trim()
                if (message.isNotEmpty()) {
                    appendLine("YOU     $message")
                    input.text.clear()
                    input.isEnabled = false
                    isEnabled = false

                    brainExecutor.execute {
                        val interaction = runCatching { coordinator.receive(message) }
                        runOnUiThread {
                            interaction.onSuccess { result ->
                                if (result.actionDecision is ActionDecision.Blocked) {
                                    appendLine("SAFETY  ${result.actionDecision.reason}")
                                }
                                appendLine("LITTLE  ${result.output.speech}")
                                appendLine(
                                    String.format(
                                        Locale.US,
                                        "PERF    brain %.2f ms | total %.2f ms",
                                        result.timing.brainMs,
                                        result.timing.totalMs
                                    )
                                )
                            }.onFailure { failure ->
                                appendLine("ERROR   ${failure.message ?: failure.javaClass.simpleName}")
                            }
                            input.isEnabled = true
                            isEnabled = true
                        }
                    }
                }
            }
        }

        val inputRow = LinearLayout(this).apply {
            orientation = LinearLayout.HORIZONTAL
            gravity = Gravity.CENTER_VERTICAL
            setPadding(16, 8, 16, 16)
            addView(input, LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.WRAP_CONTENT, 1f))
            addView(send)
        }

        val root = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setBackgroundColor(Color.rgb(17, 19, 24))
            setOnApplyWindowInsetsListener { view, insets ->
                view.setPadding(
                    0,
                    insets.systemWindowInsetTop,
                    0,
                    insets.systemWindowInsetBottom
                )
                insets
            }
            addView(transcript, LinearLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT,
                0,
                1f
            ))
            addView(inputRow)
        }

        setContentView(root)
    }

    override fun onDestroy() {
        brainExecutor.execute { runCatching { brainResource?.close() } }
        brainExecutor.shutdown()
        super.onDestroy()
    }

    private fun appendLine(line: String) {
        transcript.append("$line\n")
        transcript.post {
            val contentHeight = transcript.layout?.height ?: 0
            transcript.scrollTo(0, (contentHeight - transcript.height).coerceAtLeast(0))
        }
    }

    private companion object {
        const val MODEL_RELATIVE_PATH = "models/gemma-4-E2B-it.litertlm"
    }
}
