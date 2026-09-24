package com.littleguy.robot.core

data class BrainInput(
    val text: String,
    val recentTurns: List<String>,
    val observation: RobotObservation
)

data class BrainOutput(
    val speech: String,
    val action: RobotAction = RobotAction.None,
    val inference: InferenceMetrics? = null
)

data class InferenceMetrics(
    val sdkReportedTtftMs: Double,
    val measuredFirstTextMs: Double? = null,
    val prefillTokens: Int,
    val prefillTokensPerSecond: Double,
    val decodeTokens: Int,
    val decodeTokensPerSecond: Double
)

sealed interface RobotAction {
    data object None : RobotAction
    data class TurnToward(val degrees: Int) : RobotAction
    data class Gesture(val name: String) : RobotAction
}

fun interface BrainBackend {
    fun respond(input: BrainInput): BrainOutput
}
