package com.littleguy.robot.core

interface ConversationMemory {
    fun recentTurns(limit: Int): List<String>
    fun append(turns: List<String>)
}

class InMemoryConversationMemory(
    private val maxTurns: Int = 200
) : ConversationMemory {
    private val turns = ArrayDeque<String>()

    init {
        require(maxTurns > 0) { "maxTurns must be positive" }
    }

    @Synchronized
    override fun recentTurns(limit: Int): List<String> {
        require(limit >= 0) { "limit cannot be negative" }
        return turns.takeLast(limit)
    }

    @Synchronized
    override fun append(turns: List<String>) {
        this.turns.addAll(turns)
        while (this.turns.size > maxTurns) this.turns.removeFirst()
    }
}
