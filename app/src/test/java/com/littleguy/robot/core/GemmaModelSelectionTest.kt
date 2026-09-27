package com.littleguy.robot.core

import java.nio.file.Files
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertThrows
import org.junit.Assert.assertTrue
import org.junit.Test

class GemmaModelSelectionTest {
    @Test
    fun defaultsToE2BWithoutSelectionFile() {
        val root = Files.createTempDirectory("little-guy-model").toFile()
        try {
            val selected = GemmaModels.select(root)
            assertEquals("e2b", selected.id)
            assertEquals("gemma-4-E2B-it.litertlm", selected.file.name)
            assertFalse(selected.explicit)
        } finally {
            root.deleteRecursively()
        }
    }

    @Test
    fun explicitE4BUsesItsOwnFile() {
        val root = Files.createTempDirectory("little-guy-model").toFile()
        try {
            val selectionFile = root.resolve("models/selected-model.txt")
            root.resolve("models").mkdirs()
            selectionFile.writeText("e4b\n")
            val selected = GemmaModels.select(root)
            assertEquals("e4b", selected.id)
            assertEquals("gemma-4-E4B-it.litertlm", selected.file.name)
            assertTrue(selected.explicit)
        } finally {
            root.deleteRecursively()
        }
    }

    @Test
    fun rejectsUnknownModelInsteadOfGuessing() {
        val root = Files.createTempDirectory("little-guy-model").toFile()
        try {
            val selectionFile = root.resolve("models/selected-model.txt")
            root.resolve("models").mkdirs()
            selectionFile.writeText("e5b")
            assertThrows(IllegalArgumentException::class.java) {
                GemmaModels.select(root)
            }
        } finally {
            root.deleteRecursively()
        }
    }
}
