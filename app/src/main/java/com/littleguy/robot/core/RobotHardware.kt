package com.littleguy.robot.core

interface RobotHardware {
    fun observe(): RobotObservation
    fun perform(action: RobotAction)
}

data class RobotObservation(
    val distanceCm: Int? = null,
    val faceDetected: Boolean = false,
    val soundDirectionDegrees: Int? = null
)

class SimulatedHardware(
    private val log: (String) -> Unit
) : RobotHardware {
    override fun observe() = RobotObservation(
        distanceCm = 80,
        faceDetected = true,
        soundDirectionDegrees = -15
    )

    override fun perform(action: RobotAction) {
        when (action) {
            RobotAction.None -> Unit
            is RobotAction.Gesture -> log("MOTION  ${action.name}")
            is RobotAction.TurnToward -> log("MOTION  turn ${action.degrees} degrees")
        }
    }
}
