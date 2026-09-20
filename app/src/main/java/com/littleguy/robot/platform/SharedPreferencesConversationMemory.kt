package com.littleguy.robot.platform

import android.content.SharedPreferences
import com.littleguy.robot.core.ConversationMemory
import org.json.JSONArray

class SharedPreferencesConversationMemory(
    private val preferences: SharedPreferences,
    private val maxTurns: Int = 200
) : ConversationMemory {
    init {
        require(maxTurns > 0) { "maxTurns must be positive" }
    }

    @Synchronized
    override fun recentTurns(limit: Int): List<String> {
        require(limit >= 0) { "limit cannot be negative" }
        return readAll().takeLast(limit)
    }

    @Synchronized
    override fun append(turns: List<String>) {
        val retained = (readAll() + turns).takeLast(maxTurns)
        val encoded = JSONArray()
        retained.forEach(encoded::put)
        preferences.edit().putString(KEY_TURNS, encoded.toString()).apply()
    }

    private fun readAll(): List<String> {
        val encoded = preferences.getString(KEY_TURNS, null) ?: return emptyList()
        return runCatching {
            val array = JSONArray(encoded)
            List(array.length()) { index -> array.getString(index) }
        }.getOrDefault(emptyList())
    }

    private companion object {
        const val KEY_TURNS = "conversation_turns"
    }
}
