package com.littleguy.robot.alpha

class AlphaFrameStreamDecoder {
    private var buffered = ByteArray(0)

    fun feed(chunk: ByteArray): List<AlphaFrame> {
        if (chunk.isNotEmpty()) buffered += chunk
        val frames = mutableListOf<AlphaFrame>()

        while (true) {
            val headerIndex = findHeader(buffered)
            if (headerIndex < 0) {
                buffered = buffered.takeLastPossibleHeaderByte()
                break
            }
            if (headerIndex > 0) buffered = buffered.copyOfRange(headerIndex, buffered.size)
            if (buffered.size < LENGTH_PREFIX_BYTES) break

            val declaredLength = buffered[2].unsigned()
            if (declaredLength < MINIMUM_LENGTH_WITHOUT_END) {
                buffered = buffered.copyOfRange(1, buffered.size)
                continue
            }

            val completeFrameBytes = declaredLength + 1
            if (buffered.size < completeFrameBytes) break

            val candidate = buffered.copyOfRange(0, completeFrameBytes)
            val frame = runCatching { AlphaFrameCodec.decode(candidate) }.getOrNull()
            if (frame == null) {
                buffered = buffered.copyOfRange(1, buffered.size)
                continue
            }

            frames += frame
            buffered = buffered.copyOfRange(completeFrameBytes, buffered.size)
        }

        return frames
    }

    private fun findHeader(bytes: ByteArray): Int {
        for (index in 0 until bytes.lastIndex) {
            if (bytes[index].unsigned() == HEADER_FIRST &&
                bytes[index + 1].unsigned() == HEADER_SECOND
            ) {
                return index
            }
        }
        return -1
    }

    private fun ByteArray.takeLastPossibleHeaderByte(): ByteArray =
        if (lastOrNull()?.unsigned() == HEADER_FIRST) byteArrayOf(last()) else ByteArray(0)

    private companion object {
        const val HEADER_FIRST = 0xFB
        const val HEADER_SECOND = 0xBF
        const val LENGTH_PREFIX_BYTES = 3
        const val MINIMUM_LENGTH_WITHOUT_END = 5
    }
}
