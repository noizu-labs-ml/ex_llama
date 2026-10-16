import importlib.util
from pathlib import Path
import struct
import tempfile
import unittest
import numpy as np

spec = importlib.util.spec_from_file_location("exporter", Path(__file__).with_name("export_jev_head.py"))
exporter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(exporter)


class ExportTest(unittest.TestCase):
    def fixture(self, directory, **changes):
        arrays = {"mu": np.array([[1., 2.]], dtype=np.float32),
                  "sd": np.array([[2., 4.]], dtype=np.float32),
                  "linear.weight": np.zeros((256, 2), dtype=np.float32),
                  "linear.bias": np.arange(256, dtype=np.float32)}
        arrays.update(changes)
        source = Path(directory) / "head.npz"
        np.savez(source, **arrays)
        return source

    def test_exact_layout_and_determinism(self):
        with tempfile.TemporaryDirectory() as directory:
            source = self.fixture(directory)
            output = Path(directory) / "head.bin"
            digest = exporter.export(source, output)
            data = output.read_bytes()
            self.assertEqual(data[:16], b"JEVH0001" + struct.pack("<II", 2, 256))
            self.assertEqual(struct.unpack("<ffff", data[16:32]), (1., 2., 2., 4.))
            self.assertEqual(len(data), 16 + (4 + 256 * 2 + 256) * 4)
            self.assertEqual(struct.unpack("<f", data[-4:])[0], 255.)
            self.assertEqual(exporter.export(source, output), digest)

    def test_invalid_sd_dimensions_and_nonfinite(self):
        changes = [{"sd": np.array([[0., 1.]])},
                   {"mu": np.zeros((3,))},
                   {"linear.weight": np.zeros((255, 2))},
                   {"linear.bias": np.full(256, np.nan)}]
        with tempfile.TemporaryDirectory() as directory:
            for change in changes:
                with self.subTest(change=list(change)):
                    source = self.fixture(directory, **change)
                    with self.assertRaises(ValueError):
                        exporter.export(source, Path(directory) / "head.bin")

    def test_npz_plain_weight_bias_aliases(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "head.npz"
            np.savez(source, mu=np.zeros(2), sd=np.ones(2),
                     weight=np.zeros((256, 2)), bias=np.zeros(256))
            self.assertEqual(len(exporter.export(source, Path(directory) / "head.bin")), 64)


if __name__ == "__main__":
    unittest.main()
