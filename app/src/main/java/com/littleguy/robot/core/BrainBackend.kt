package com.littleguy.robot.core

data class BrainInput(
    val text: String,
    val recentTurns: List<String>,
    val observation: RobotObservation
)

data class BrainOutput(
    val speech: String,
    val action: RobotAction = RobotAction.None
)

sealed interface RobotAction {
    data object None : RobotAction
    data class TurnToward(val degrees: Int) : RobotAction
    data class Gesture(val name: String) : RobotAction
}

fun interface BrainBackend {
    fun respond(input: BrainInput): BrainOutput
}
