package com.littleguy.robot

import android.content.BroadcastReceiver
import android.content.ComponentName
import android.content.Context
import android.content.Intent
import android.content.ServiceConnection
import android.os.IBinder
import android.util.Base64
import android.util.Log

/** ADB-shell prompt entry point for debug builds only. */
class DebugPromptReceiver : BroadcastReceiver() {
    override fun onReceive(context: Context, intent: Intent) {
        val prompt = intent.getStringExtra("prompt")?.trim().orEmpty()
        val requestId = intent.getStringExtra("request_id")?.trim().orEmpty()
        if (prompt.isEmpty() || prompt.length > 500 || requestId.isEmpty()) {
            Log.w(TAG, "$requestId ERROR invalid prompt or request ID")
            return
        }

        val app = context.applicationContext
        Log.i(TAG, "$requestId START")
        val connection = object : ServiceConnection {
            private var binder: LittleGuyService.LocalBinder? = null
            private var observer: LittleGuyService.Observer? = null
            private var submitted = false
            private var initialLines = 0

            override fun onServiceConnected(name: ComponentName, service: IBinder) {
                val runtime = service as LittleGuyService.LocalBinder
                binder = runtime
                val listener = LittleGuyService.Observer { snapshot ->
                    if (!submitted) {
                        initialLines = snapshot.lines.size
                    } else if (!snapshot.busy) {
                        snapshot.lines.drop(initialLines).forEach { line ->
                            Log.i(TAG, "$requestId $line")
                        }
                        Log.i(TAG, "$requestId DONE")
                        disconnect()
                    }
                }
                observer = listener
                runtime.observe(listener)
                submitted = runtime.submit(prompt) { chunk ->
                    val encoded = Base64.encodeToString(
                        chunk.toByteArray(Charsets.UTF_8),
                        Base64.NO_WRAP
                    )
                    Log.i(TAG, "$requestId CHUNK64 $encoded")
                }
                if (!submitted) {
                    Log.w(TAG, "$requestId ERROR runtime busy or unavailable")
                    disconnect()
                }
            }

            override fun onServiceDisconnected(name: ComponentName) {
                Log.w(TAG, "$requestId ERROR runtime disconnected")
                disconnect()
            }

            private fun disconnect() {
                observer?.let { binder?.removeObserver(it) }
                observer = null
                binder = null
                runCatching { app.unbindService(this) }
            }
        }

        if (!app.bindService(
                Intent(app, LittleGuyService::class.java),
                connection,
                Context.BIND_AUTO_CREATE
            )
        ) {
            Log.w(TAG, "$requestId ERROR service binding failed")
        }
    }

    private companion object {
        const val TAG = "LittleGuyDebug"
    }
}
