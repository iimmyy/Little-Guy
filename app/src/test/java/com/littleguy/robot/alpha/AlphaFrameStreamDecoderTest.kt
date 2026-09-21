package com.littleguy.robot.alpha

import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class AlphaFrameStreamDecoderTest {
    @Test
    fun recoversSplitFramesAndSkipsLeadingNoise() {
        val decoder = AlphaFrameStreamDecoder()
        val heartbeat = AlphaProtocol.heartbeat()
        val battery = AlphaProtocol.readBattery()

        assertTrue(decoder.feed(bytes(0x44, 0x55, 0xFB)).isEmpty())

        val frames = decoder.feed(
            heartbeat.copyOfRange(1, heartbeat.size) + battery
        )

        assertEquals(2, frames.size)
        assertEquals(AlphaCommand.HEARTBEAT.wireValue, frames[0].command)
        assertEquals(AlphaCommand.READ_BATTERY.wireValue, frames[1].command)
    }

    @Test
    fun fakeTransportCopiesWritesAndReturnsReadsInOrder() {
        val firstRead = AlphaProtocol.handshake()
        val secondRead = AlphaProtocol.readBattery()
        val transport = FakeAlphaTransport(listOf(firstRead))
        val write = AlphaProtocol.heartbeat()

        transport.write(write)
        write[0] = 0x00
        transport.enqueueRead(secondRead)

        assertArrayEquals(AlphaProtocol.heartbeat(), transport.writes().single())
        assertArrayEquals(firstRead, transport.read())
        assertArrayEquals(secondRead, transport.read())
        assertEquals(null, transport.read())
    }

    private fun bytes(vararg values: Int) = ByteArray(values.size) { values[it].toByte() }
}
