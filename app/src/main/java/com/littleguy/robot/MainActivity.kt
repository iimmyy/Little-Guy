package com.littleguy.robot

import android.app.Activity
import android.content.ComponentName
import android.content.Context
import android.content.Intent
import android.content.ServiceConnection
import android.content.res.ColorStateList
import android.graphics.Color
import android.os.Bundle
import android.os.IBinder
import android.text.method.ScrollingMovementMethod
import android.view.Gravity
import android.view.ViewGroup
import android.widget.Button
import android.widget.EditText
import android.widget.LinearLayout
import android.widget.TextView

class MainActivity : Activity() {
    private lateinit var transcript: TextView
    private lateinit var input: EditText
    private lateinit var send: Button
    private var runtime: LittleGuyService.LocalBinder? = null
    private var isBound = false

    private val observer = LittleGuyService.Observer(::render)

    private val serviceConnection = object : ServiceConnection {
        override fun onServiceConnected(name: ComponentName, service: IBinder) {
            runtime = service as LittleGuyService.LocalBinder
            isBound = true
            runtime?.observe(observer)
        }

        override fun onServiceDisconnected(name: ComponentName) {
            runtime = null
            isBound = false
            setControlsEnabled(false)
            transcript.append("SYSTEM  Runtime disconnected. Waiting for Android to reconnect.\n")
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        transcript = TextView(this).apply {
            setTextColor(Color.rgb(225, 229, 238))
            textSize = 16f
            movementMethod = ScrollingMovementMethod()
            setPadding(32, 24, 32, 24)
            text = "SYSTEM  Connecting to Little Guy's runtime…\n"
        }

        input = EditText(this).apply {
            hint = "Say something to Little Guy"
            setSingleLine(true)
            setTextColor(Color.WHITE)
            setHintTextColor(Color.rgb(165, 170, 180))
            backgroundTintList = ColorStateList.valueOf(Color.rgb(92, 107, 192))
            isEnabled = false
        }

        send = Button(this).apply {
            text = "Send"
            setTextColor(Color.WHITE)
            backgroundTintList = ColorStateList.valueOf(Color.rgb(73, 88, 180))
            isEnabled = false
            setOnClickListener {
                val message = input.text.toString().trim()
                if (message.isNotEmpty() && runtime?.submit(message) == true) {
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
            fitsSystemWindows = true
            addView(
                transcript,
                LinearLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, 0, 1f)
            )
            addView(inputRow)
        }

        setContentView(root)
        startRuntimeService()
    }

    override fun onStart() {
        super.onStart()
        bindService(
            Intent(this, LittleGuyService::class.java),
            serviceConnection,
            Context.BIND_AUTO_CREATE
        )
    }

    override fun onStop() {
        if (isBound) {
            runtime?.removeObserver(observer)
            unbindService(serviceConnection)
            runtime = null
            isBound = false
        }
        super.onStop()
    }

    private fun startRuntimeService() {
        startForegroundService(Intent(this, LittleGuyService::class.java))
    }

    private fun render(snapshot: LittleGuyService.RuntimeSnapshot) {
        transcript.text = snapshot.lines.joinToString(separator = "\n", postfix = "\n")
        setControlsEnabled(!snapshot.busy)
        transcript.post {
            val contentHeight = transcript.layout?.height ?: 0
            transcript.scrollTo(0, (contentHeight - transcript.height).coerceAtLeast(0))
        }
    }

    private fun setControlsEnabled(enabled: Boolean) {
        input.isEnabled = enabled
        send.isEnabled = enabled
    }
}
