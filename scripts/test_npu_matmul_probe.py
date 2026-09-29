import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np


SCRIPT = Path(__file__).with_name("npu-matmul-probe.py")
SPEC = importlib.util.spec_from_file_location("npu_matmul_probe", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class OnGridWeightsTest(unittest.TestCase):
    def test_weights_are_exact_int4_multiples_with_pinned_extremes(self):
        weights, scale = MODULE.on_grid_weights(16, 5, np.random.default_rng(1))
        q = weights / scale
        np.testing.assert_allclose(q, np.round(q), atol=1e-4)
        self.assertLessEqual(np.abs(np.round(q)).max(), 7)
        np.testing.assert_allclose(np.abs(weights).max(axis=0), 7 * scale, rtol=1e-6)

    def test_int4_and_int8_encodings_share_scale_and_range(self):
        _, scale = MODULE.on_grid_weights(16, 3, np.random.default_rng(2))
        int4 = MODULE.weight_encodings(scale, 4)
        int8 = MODULE.weight_encodings(scale, 8)
        self.assertEqual(len(int4), 3)
        for a, b in zip(int4, int8):
            self.assertEqual((a["bitwidth"], b["bitwidth"]), (4, 8))
            self.assertEqual((a["scale"], a["min"], a["max"]), (b["scale"], b["min"], b["max"]))
            self.assertEqual((a["offset"], b["offset"]), (-8, -128))


class CompareTest(unittest.TestCase):
    def test_identical_output_passes(self):
        ref = np.arange(1, 9, dtype=np.float32)
        result = MODULE.compare(ref, ref.copy())
        self.assertAlmostEqual(result["cosine"], 1.0)
        self.assertEqual(result["max_abs_error"], 0.0)

    def test_size_mismatch_is_an_error(self):
        with self.assertRaises(ValueError):
            MODULE.compare(np.zeros(4, np.float32), np.zeros(5, np.float32))

    def test_degenerate_output_fails(self):
        ref = np.linspace(-1, 1, 64, dtype=np.float32)
        self.assertLess(MODULE.compare(ref, np.zeros_like(ref))["cosine"], 0.99)


@unittest.skipUnless(importlib.util.find_spec("onnx"), "onnx not installed")
class BuildTest(unittest.TestCase):
    def test_build_writes_consistent_probe(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            MODULE.build(out, k=32, n=24, rows=2, samples=3, seed=3)
            lines = (out / "input_list.txt").read_text().split()
            self.assertEqual(lines, [f"inputs/x_{i}.raw" for i in range(3)])
            x = np.fromfile(out / lines[0], np.float32).reshape(2, 32)
            y = np.fromfile(out / "reference/y_0.raw", np.float32).reshape(2, 24)

            import onnx
            from onnx import numpy_helper
            model = onnx.load(out / "matmul.onnx")
            weights = numpy_helper.to_array(model.graph.initializer[0])
            np.testing.assert_allclose(x @ weights, y, rtol=1e-5, atol=1e-5)

            overrides = json.loads((out / "overrides_int4.json").read_text())
            self.assertEqual(len(overrides["param_encodings"]["w"]), 24)
            self.assertEqual(set(overrides["activation_encodings"]), {"x", "y"})


if __name__ == "__main__":
    unittest.main()
