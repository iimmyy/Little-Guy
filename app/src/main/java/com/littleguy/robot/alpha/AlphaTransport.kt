package com.littleguy.robot.alpha

interface AlphaTransport : AutoCloseable {
    fun write(bytes: ByteArray)
    fun read(): ByteArray?
    override fun close() = Unit
}

class FakeAlphaTransport(
    initialReads: List<ByteArray> = emptyList()
) : AlphaTransport {
    private val pendingReads = ArrayDeque(initialReads.map(ByteArray::copyOf))
    private val recordedWrites = mutableListOf<ByteArray>()

    @Synchronized
    override fun write(bytes: ByteArray) {
        recordedWrites += bytes.copyOf()
    }

    @Synchronized
    override fun read(): ByteArray? = pendingReads.removeFirstOrNull()?.copyOf()

    @Synchronized
    fun enqueueRead(bytes: ByteArray) {
        pendingReads += bytes.copyOf()
    }

    @Synchronized
    fun writes(): List<ByteArray> = recordedWrites.map(ByteArray::copyOf)
}
