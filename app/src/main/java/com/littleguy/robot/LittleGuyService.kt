package com.littleguy.robot

import android.app.Notification
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.PendingIntent
import android.app.Service
import android.content.Intent
import android.content.pm.ApplicationInfo
import android.content.pm.ServiceInfo
import android.os.Binder
import android.os.Build
import android.os.Handler
import android.os.IBinder
import android.os.Looper
import com.google.ai.edge.litertlm.Backend
import com.littleguy.robot.core.ActionDecision
import com.littleguy.robot.core.BrainBackend
import com.littleguy.robot.core.DevelopmentBrain
import com.littleguy.robot.core.LiteRtLmBrain
import com.littleguy.robot.core.RobotCoordinator
import com.littleguy.robot.core.SimulatedHardware
import com.littleguy.robot.platform.SharedPreferencesConversationMemory
import java.io.File
import java.util.Locale
import java.util.concurrent.CopyOnWriteArraySet
import java.util.concurrent.Executors

class LittleGuyService : Service() {
    data class RuntimeSnapshot(
        val lines: List<String>,
        val busy: Boolean
    )

    fun interface Observer {
        fun onSnapshot(snapshot: RuntimeSnapshot)
    }

    inner class LocalBinder : Binder() {
        fun observe(observer: Observer) {
            observers += observer
            observer.onSnapshot(snapshot())
        }

        fun removeObserver(observer: Observer) {
            observers -= observer
        }

        fun submit(text: String): Boolean = submitMessage(text)
    }

    private val binder = LocalBinder()
    private val mainHandler = Handler(Looper.getMainLooper())
    private val brainExecutor = Executors.newSingleThreadExecutor()
    private val observers = CopyOnWriteArraySet<Observer>()
    private val transcript = ArrayDeque<String>()
    private lateinit var coordinator: RobotCoordinator
    private var brainResource: AutoCloseable? = null
    private var busy = false
    private var shuttingDown = false

    override fun onCreate() {
        super.onCreate()
        val notification = createNotification()
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.UPSIDE_DOWN_CAKE) {
            startForeground(
                NOTIFICATION_ID,
                notification,
                ServiceInfo.FOREGROUND_SERVICE_TYPE_SPECIAL_USE
            )
        } else {
            startForeground(NOTIFICATION_ID, notification)
        }

        val modelFile = File(getExternalFilesDir(null) ?: filesDir, MODEL_RELATIVE_PATH)
        val brain: BrainBackend
        val brainStatus: String
        if (modelFile.isFile) {
            val gemmaBrain = LiteRtLmBrain(
                modelPath = modelFile.absolutePath,
                cacheDir = cacheDir.absolutePath,
                backend = Backend.GPU(),
                streamForMeasurement = applicationInfo.flags and ApplicationInfo.FLAG_DEBUGGABLE != 0
            )
            brain = gemmaBrain
            brainResource = gemmaBrain
            brainStatus = "Gemma model found; it will load on the first message using the device GPU."
        } else {
            brain = DevelopmentBrain()
            brainStatus = "Development brain online. Gemma model not found at:\n${modelFile.absolutePath}"
        }

        coordinator = RobotCoordinator(
            brain = brain,
            hardware = SimulatedHardware(::appendLine),
            memory = SharedPreferencesConversationMemory(
                getSharedPreferences("little_guy_memory", MODE_PRIVATE)
            )
        )
        appendLine("SYSTEM  $brainStatus")
    }

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int = START_STICKY

    override fun onBind(intent: Intent): IBinder = binder

    override fun onDestroy() {
        shuttingDown = true
        observers.clear()
        brainExecutor.execute { runCatching { brainResource?.close() } }
        brainExecutor.shutdown()
        super.onDestroy()
    }

    private fun submitMessage(rawText: String): Boolean {
        check(Looper.myLooper() == Looper.getMainLooper()) {
            "Runtime commands must be submitted from the main thread."
        }
        val text = rawText.trim()
        if (text.isEmpty() || busy || shuttingDown) return false

        busy = true
        appendLine("YOU     $text")
        publish()

        brainExecutor.execute {
            val interaction = runCatching { coordinator.receive(text) }
            mainHandler.post {
                if (shuttingDown) return@post
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
                    result.output.inference?.let { metrics ->
                        metrics.measuredFirstTextMs?.let { firstTextMs ->
                            appendLine(
                                String.format(Locale.US, "MODEL   first streamed text %.0f ms", firstTextMs)
                            )
                        }
                        appendLine(
                            String.format(
                                Locale.US,
                                "MODEL   SDK TTFT %.0f ms (unverified) | prefill %d @ %.1f tok/s | decode %d @ %.1f tok/s",
                                metrics.sdkReportedTtftMs,
                                metrics.prefillTokens,
                                metrics.prefillTokensPerSecond,
                                metrics.decodeTokens,
                                metrics.decodeTokensPerSecond
                            )
                        )
                    }
                }.onFailure { failure ->
                    appendLine("ERROR   ${failure.message ?: failure.javaClass.simpleName}")
                }
                busy = false
                publish()
            }
        }
        return true
    }

    private fun appendLine(line: String) {
        if (Looper.myLooper() != Looper.getMainLooper()) {
            mainHandler.post { appendLine(line) }
            return
        }
        transcript += line
        while (transcript.size > MAX_TRANSCRIPT_LINES) transcript.removeFirst()
    }

    private fun publish() {
        val snapshot = snapshot()
        observers.forEach { it.onSnapshot(snapshot) }
    }

    private fun snapshot() = RuntimeSnapshot(
        lines = transcript.toList(),
        busy = busy
    )

    private fun createNotification(): Notification {
        val manager = getSystemService(NotificationManager::class.java)
        manager.createNotificationChannel(
            NotificationChannel(
                NOTIFICATION_CHANNEL_ID,
                "Little Guy runtime",
                NotificationManager.IMPORTANCE_LOW
            )
        )

        val dashboardIntent = PendingIntent.getActivity(
            this,
            0,
            Intent(this, MainActivity::class.java),
            PendingIntent.FLAG_IMMUTABLE or PendingIntent.FLAG_UPDATE_CURRENT
        )
        return Notification.Builder(this, NOTIFICATION_CHANNEL_ID)
            .setSmallIcon(R.drawable.ic_little_guy_status)
            .setContentTitle("Little Guy is awake")
            .setContentText("Robot runtime active")
            .setContentIntent(dashboardIntent)
            .setOngoing(true)
            .build()
    }

    private companion object {
        const val MODEL_RELATIVE_PATH = "models/gemma-4-E2B-it.litertlm"
        const val MAX_TRANSCRIPT_LINES = 250
        const val NOTIFICATION_ID = 1001
        const val NOTIFICATION_CHANNEL_ID = "little_guy_runtime"
    }
}
