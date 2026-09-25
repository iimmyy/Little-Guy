import importlib.util
from pathlib import Path
import tempfile
import unittest


SCRIPT = Path(__file__).with_name("alpha-hts-inspect.py")
SPEC = importlib.util.spec_from_file_location("alpha_hts_inspect", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class HtsInspectTest(unittest.TestCase):
    def test_decodes_pose(self):
        pose = bytearray(33)
        pose[:2] = b"\xfb\xbf"
        pose[8:24] = bytes(range(1, 17))
        pose[28] = 10
        pose[29:31] = (8).to_bytes(2, "big")
        pose[-1] = 0xED
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sample.hts"
            path.write_bytes(bytes(33) + pose + bytes(33))
            self.assertEqual(list(MODULE.decode(path)), [(1, 200, 200, *range(1, 17))])

    def test_rejects_bad_marker(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.hts"
            path.write_bytes(bytes(99))
            with self.assertRaisesRegex(ValueError, "invalid frame markers"):
                list(MODULE.decode(path))


if __name__ == "__main__":
    unittest.main()
