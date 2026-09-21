package com.littleguy.robot.alpha

import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertEquals
import org.junit.Assert.assertThrows
import org.junit.Test

class AlphaFrameCodecTest {
    @Test
    fun encodesHeartbeatFixtureFromPublishedFrameRules() {
        assertArrayEquals(
            bytes(0xFB, 0xBF, 0x06, 0x08, 0x00, 0x0E, 0xED),
            AlphaProtocol.heartbeat()
        )
    }

    @Test
    fun decodesAValidFrame() {
        val frame = AlphaFrameCodec.decode(
            bytes(0xFB, 0xBF, 0x06, 0x18, 0x00, 0x1E, 0xED)
        )

        assertEquals(AlphaCommand.READ_BATTERY.wireValue, frame.command)
        assertArrayEquals(bytes(0x00), frame.parameters)
    }

    @Test
    fun rejectsCorruptLengthChecksumAndEndMarker() {
        val valid = AlphaProtocol.handshake()

        assertThrows(AlphaProtocolException::class.java) {
            AlphaFrameCodec.decode(valid.copyOf().also { it[2] = 0x07 })
        }
        assertThrows(AlphaProtocolException::class.java) {
            AlphaFrameCodec.decode(valid.copyOf().also { it[it.lastIndex - 1] = 0x00 })
        }
        assertThrows(AlphaProtocolException::class.java) {
            AlphaFrameCodec.decode(valid.copyOf().also { it[it.lastIndex] = 0x00 })
        }
    }

    @Test
    fun validatesServoIdBeforeBuildingAReadRequest() {
        assertThrows(IllegalArgumentException::class.java) {
            AlphaProtocol.readSingleAngle(0)
        }
        assertThrows(IllegalArgumentException::class.java) {
            AlphaProtocol.readSingleAngle(17)
        }
    }

    private fun bytes(vararg values: Int) = ByteArray(values.size) { values[it].toByte() }
}
