package com.littleguy.robot.core

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class RobotCoordinatorTest {
    @Test
    fun routesBrainActionToHardware() {
        val actions = mutableListOf<RobotAction>()
        val hardware = object : RobotHardware {
            override fun observe() = RobotObservation(distanceCm = 42)

            override fun perform(action: RobotAction) {
                actions += action
            }
        }
        val brain = BrainBackend { input ->
            BrainOutput(
                speech = "Object at ${input.observation.distanceCm} centimetres.",
                action = RobotAction.TurnToward(20)
            )
        }

        val output = RobotCoordinator(brain, hardware).receive("look")

        assertEquals("Object at 42 centimetres.", output.speech)
        assertEquals(listOf(RobotAction.TurnToward(20)), actions)
    }

    @Test
    fun developmentBrainUsesSimulatedSensorData() {
        val output = DevelopmentBrain().respond(
            BrainInput(
                text = "How close is it?",
                recentTurns = emptyList(),
                observation = RobotObservation(distanceCm = 80)
            )
        )

        assertTrue(output.speech.contains("80 centimetres"))
    }
}
