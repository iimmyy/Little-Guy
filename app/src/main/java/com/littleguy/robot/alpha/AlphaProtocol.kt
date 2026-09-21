package com.littleguy.robot.alpha

enum class AlphaCommand(val wireValue: Int) {
    HANDSHAKE(0x01),
    HEARTBEAT(0x08),
    READ_STATE(0x0A),
    READ_SOFTWARE_VERSION(0x11),
    READ_BATTERY(0x18),
    READ_HARDWARE_VERSION(0x20),
    READ_SINGLE_ANGLE(0x24),
    READ_ALL_ANGLES(0x25)
}

object AlphaProtocol {
    fun handshake() = request(AlphaCommand.HANDSHAKE)

    fun heartbeat() = request(AlphaCommand.HEARTBEAT)

    fun readState() = request(AlphaCommand.READ_STATE)

    fun readSoftwareVersion() = request(AlphaCommand.READ_SOFTWARE_VERSION)

    fun readBattery() = request(AlphaCommand.READ_BATTERY)

    fun readHardwareVersion() = request(AlphaCommand.READ_HARDWARE_VERSION)

    fun readSingleAngle(servoId: Int): ByteArray {
        require(servoId in 1..SERVO_COUNT) { "servoId must be between 1 and $SERVO_COUNT" }
        return request(AlphaCommand.READ_SINGLE_ANGLE, byteArrayOf(servoId.toByte()))
    }

    fun readAllAngles() = request(AlphaCommand.READ_ALL_ANGLES)

    private fun request(
        command: AlphaCommand,
        parameters: ByteArray = byteArrayOf(0x00)
    ): ByteArray = AlphaFrameCodec.encode(AlphaFrame(command.wireValue, parameters))

    private const val SERVO_COUNT = 16
}
