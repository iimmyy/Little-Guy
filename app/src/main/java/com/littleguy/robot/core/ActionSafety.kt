package com.littleguy.robot.core

data class ActionPolicy(
    val maxTurnDegrees: Int = 45,
    val allowedGestures: Set<String> = setOf("wave", "dance")
) {
    init {
        require(maxTurnDegrees > 0) { "maxTurnDegrees must be positive" }
    }
}

sealed interface ActionDecision {
    data class Allowed(val action: RobotAction) : ActionDecision
    data class Blocked(val action: RobotAction, val reason: String) : ActionDecision
}

class ActionSafetyGate(
    private val policy: ActionPolicy = ActionPolicy()
) {
    fun evaluate(action: RobotAction): ActionDecision = when (action) {
        RobotAction.None -> ActionDecision.Allowed(action)

        is RobotAction.TurnToward -> {
            if (action.degrees in -policy.maxTurnDegrees..policy.maxTurnDegrees) {
                ActionDecision.Allowed(action)
            } else {
                ActionDecision.Blocked(
                    action,
                    "turn must stay within ${policy.maxTurnDegrees} degrees"
                )
            }
        }

        is RobotAction.Gesture -> {
            if (action.name in policy.allowedGestures) {
                ActionDecision.Allowed(action)
            } else {
                ActionDecision.Blocked(action, "gesture is not approved")
            }
        }
    }
}
