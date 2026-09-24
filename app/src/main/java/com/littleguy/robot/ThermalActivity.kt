package com.littleguy.robot

import android.app.Activity
import android.content.Intent
import android.content.IntentFilter
import android.graphics.Color
import android.os.BatteryManager
import android.os.Bundle
import android.os.Build
import android.os.Handler
import android.os.Looper
import android.os.PowerManager
import android.text.method.ScrollingMovementMethod
import android.graphics.Typeface
import android.widget.TextView
import java.io.File
import java.util.Locale

/** A read-only, on-device view for Q6A bring-up. */
class ThermalActivity : Activity() {
    private val handler = Handler(Looper.getMainLooper())
    private lateinit var display: TextView
    private val refresh = object : Runnable {
        override fun run() {
            display.text = snapshot()
            handler.postDelayed(this, 2_000)
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        display = TextView(this).apply {
            setBackgroundColor(Color.rgb(12, 17, 22))
            setTextColor(Color.rgb(221, 239, 233))
            typeface = Typeface.MONOSPACE
            textSize = 18f
            setPadding(32, 28, 32, 28)
            movementMethod = ScrollingMovementMethod()
        }
        setContentView(display)
    }

    override fun onStart() {
        super.onStart()
        handler.post(refresh)
    }

    override fun onStop() {
        handler.removeCallbacks(refresh)
        super.onStop()
    }

    private fun snapshot(): String {
        val power = getSystemService(POWER_SERVICE) as PowerManager
        val thermalStatus = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
            power.currentThermalStatus.toString()
        } else {
            "unavailable"
        }
        val battery = registerReceiver(null, IntentFilter(Intent.ACTION_BATTERY_CHANGED))
        val batteryTenths = battery?.getIntExtra(BatteryManager.EXTRA_TEMPERATURE, -1) ?: -1
        val zones = File("/sys/class/thermal").listFiles()
            .orEmpty()
            .filter { it.name.startsWith("thermal_zone") }
            .mapNotNull { zone ->
                try {
                    val name = File(zone, "type").readText().trim()
                    val celsius = File(zone, "temp").readText().trim().toDouble() / 1000.0
                    if (celsius in 10.0..120.0) name to celsius else null
                } catch (_: Exception) {
                    null
                }
            }
            .sortedByDescending { it.second }

        return buildString {
            appendLine("Q6A INTERNALS   live / 2 sec")
            appendLine("Android thermal status: $thermalStatus")
            if (batteryTenths > 0) {
                appendLine("Battery: %.1f C".format(Locale.US, batteryTenths / 10.0))
            }
            appendLine("Thermal zones readable by app: ${zones.size}")
            appendLine()
            if (zones.isEmpty()) {
                appendLine("Android has not granted this app access to raw sensors.")
                appendLine("The Mac ADB monitor can still read them.")
            } else {
                zones.forEach { (name, value) ->
                    appendLine("%-30s %5.1f C".format(Locale.US, name, value))
                }
            }
        }
    }
}
