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
import com.littleguy.robot.core.DevelopmentBrain
import com.littleguy.robot.core.RobotCoordinator
import com.littleguy.robot.core.SimulatedHardware
import com.littleguy.robot.platform.SharedPreferencesConversationMemory
import java.util.Locale

class MainActivity : Activity() {
    private lateinit var transcript: TextView
    private lateinit var coordinator: RobotCoordinator

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        transcript = TextView(this).apply {
            setTextColor(Color.rgb(225, 229, 238))
            textSize = 16f
            movementMethod = ScrollingMovementMethod()
            setPadding(32, 24, 32, 24)
            text = "SYSTEM  Development brain online\n\n"
        }

        coordinator = RobotCoordinator(
            brain = DevelopmentBrain(),
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
                    val result = coordinator.receive(message)
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
                    input.text.clear()
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

    private fun appendLine(line: String) {
        transcript.append("$line\n")
        transcript.post {
            val contentHeight = transcript.layout?.height ?: 0
            transcript.scrollTo(0, (contentHeight - transcript.height).coerceAtLeast(0))
        }
    }
}
