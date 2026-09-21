package com.littleguy.robot.alpha

class AlphaProtocolException(message: String) : IllegalArgumentException(message)

class AlphaFrame(
    val command: Int,
    parameters: ByteArray
) {
    private val parameterBytes = parameters.copyOf()
    val parameters: ByteArray
        get() = parameterBytes.copyOf()

    init {
        require(command in BYTE_RANGE) { "command must fit in one byte" }
    }

    private companion object {
        val BYTE_RANGE = 0..0xFF
    }
}

object AlphaFrameCodec {
    private const val HEADER_FIRST = 0xFB
    private const val HEADER_SECOND = 0xBF
    private const val END = 0xED
    private const val FRAME_OVERHEAD_WITHOUT_END = 5
    private const val MINIMUM_FRAME_BYTES = 6

    fun encode(frame: AlphaFrame): ByteArray {
        val parameters = frame.parameters
        val length = parameters.size + FRAME_OVERHEAD_WITHOUT_END
        require(length <= 0xFF) { "frame is too large for the one-byte length field" }

        val checksum = checksum(length, frame.command, parameters)
        return byteArrayOf(
            HEADER_FIRST.toByte(),
            HEADER_SECOND.toByte(),
            length.toByte(),
            frame.command.toByte()
        ) + parameters + byteArrayOf(checksum.toByte(), END.toByte())
    }

    fun decode(bytes: ByteArray): AlphaFrame {
        if (bytes.size < MINIMUM_FRAME_BYTES) {
            throw AlphaProtocolException("frame is shorter than $MINIMUM_FRAME_BYTES bytes")
        }
        if (bytes[0].unsigned() != HEADER_FIRST || bytes[1].unsigned() != HEADER_SECOND) {
            throw AlphaProtocolException("frame header must be FB BF")
        }
        if (bytes.last().unsigned() != END) {
            throw AlphaProtocolException("frame must end with ED")
        }

        val declaredLength = bytes[2].unsigned()
        val actualLengthWithoutEnd = bytes.size - 1
        if (declaredLength != actualLengthWithoutEnd) {
            throw AlphaProtocolException(
                "declared length $declaredLength does not match $actualLengthWithoutEnd bytes"
            )
        }

        val command = bytes[3].unsigned()
        val parameters = bytes.copyOfRange(4, bytes.size - 2)
        val expectedChecksum = checksum(declaredLength, command, parameters)
        val actualChecksum = bytes[bytes.size - 2].unsigned()
        if (actualChecksum != expectedChecksum) {
            throw AlphaProtocolException(
                "checksum $actualChecksum does not match expected $expectedChecksum"
            )
        }

        return AlphaFrame(command, parameters)
    }

    private fun checksum(length: Int, command: Int, parameters: ByteArray): Int =
        (length + command + parameters.sumOf(Byte::unsigned)) and 0xFF
}

internal fun Byte.unsigned(): Int = toInt() and 0xFF
