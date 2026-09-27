package com.littleguy.robot.core

import java.io.File

data class GemmaModelSelection(
    val id: String,
    val file: File,
    val explicit: Boolean
)

object GemmaModels {
    private const val SELECTION_PATH = "models/selected-model.txt"

    fun select(appFilesDir: File): GemmaModelSelection {
        val selectionFile = File(appFilesDir, SELECTION_PATH)
        val explicit = selectionFile.exists()
        val id = if (explicit) selectionFile.readText().trim() else "e2b"
        val fileName = when (id) {
            "e2b" -> "gemma-4-E2B-it.litertlm"
            "e4b" -> "gemma-4-E4B-it.litertlm"
            else -> throw IllegalArgumentException(
                "Unknown model '$id' in ${selectionFile.absolutePath}; use e2b or e4b."
            )
        }
        return GemmaModelSelection(id, File(appFilesDir, "models/$fileName"), explicit)
    }
}
