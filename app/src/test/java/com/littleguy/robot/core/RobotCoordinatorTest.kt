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

        val result = RobotCoordinator(brain, hardware).receive("look")

        assertEquals("Object at 42 centimetres.", result.output.speech)
        assertEquals(listOf(RobotAction.TurnToward(20)), actions)
        assertTrue(result.actionDecision is ActionDecision.Allowed)
        assertTrue(result.timing.brainMs >= 0)
        assertTrue(result.timing.totalMs >= result.timing.brainMs)
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

    @Test
    fun developmentBrainCanReadPriorHumanTurn() {
        val output = DevelopmentBrain().respond(
            BrainInput(
                text = "What did I say?",
                recentTurns = listOf(
                    "Human: hello there",
                    "Little Guy: Hey."
                ),
                observation = RobotObservation()
            )
        )

        assertEquals("You previously said: hello there", output.speech)
    }

    @Test
    fun blocksUnsafeActionsBeforeTheyReachHardware() {
        val actions = mutableListOf<RobotAction>()
        val hardware = object : RobotHardware {
            override fun observe() = RobotObservation()

            override fun perform(action: RobotAction) {
                actions += action
            }
        }
        val brain = BrainBackend {
            BrainOutput("Trying it.", RobotAction.TurnToward(180))
        }

        val result = RobotCoordinator(brain, hardware).receive("turn around")

        assertTrue(actions.isEmpty())
        assertTrue(result.actionDecision is ActionDecision.Blocked)
    }

    @Test
    fun keepsConversationHistoryOutsideTheBrainBackend() {
        val memory = InMemoryConversationMemory()
        val observedHistory = mutableListOf<List<String>>()
        val brain = BrainBackend { input ->
            observedHistory += input.recentTurns
            BrainOutput("Okay.")
        }
        val hardware = object : RobotHardware {
            override fun observe() = RobotObservation()
            override fun perform(action: RobotAction) = Unit
        }
        val coordinator = RobotCoordinator(brain, hardware, memory)

        coordinator.receive("first")
        coordinator.receive("second")

        assertTrue(observedHistory.first().isEmpty())
        assertEquals(
            listOf("Human: first", "Little Guy: Okay."),
            observedHistory.last()
        )
    }

    @Test
    fun forwardsTextChunksWithoutTreatingThemAsActions() {
        val received = mutableListOf<String>()
        val performed = mutableListOf<RobotAction>()
        val brain = BrainBackend { input ->
            input.onTextChunk?.invoke("Hello")
            input.onTextChunk?.invoke(" there")
            BrainOutput("Hello there")
        }
        val hardware = object : RobotHardware {
            override fun observe() = RobotObservation()
            override fun perform(action: RobotAction) {
                performed += action
            }
        }

        val result = RobotCoordinator(brain, hardware).receive("hi") { chunk ->
            received += chunk
        }

        assertEquals(listOf("Hello", " there"), received)
        assertEquals("Hello there", result.output.speech)
        assertEquals(listOf(RobotAction.None), performed)
    }
}
