import hashlib
import importlib.util
import json
import math
from pathlib import Path
import struct
import tempfile
import unittest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "prepare_data.py"
spec = importlib.util.spec_from_file_location("prepare_data", SCRIPT)
data = importlib.util.module_from_spec(spec)
spec.loader.exec_module(data)


class DataTests(unittest.TestCase):
    def prepare(self, output, *source):
        data.main([*(source or ("--synthetic",)), "--database-size", "12", "--tuning-size", "3",
                   "--test-size", "4", "--dimensions", "3", "--output", str(output)])

    def check_splits(self, folder):
        manifest = json.loads((folder / "manifest.json").read_text())
        ids = []
        for name, split in manifest["splits"].items():
            ids.extend(split["source_row_ids"])
            payload = (folder / split["file"]).read_bytes()
            self.assertEqual(hashlib.sha256(payload).hexdigest(), split["sha256"])
            self.assertEqual(payload[:8], b"CVEC0001")
            rows, dimensions = struct.unpack("<II", payload[8:16])
            self.assertEqual(rows, split["rows"])
            self.assertEqual(dimensions, 3)
            values = struct.unpack(f"<{rows * dimensions}f", payload[16:])
            for i in range(rows):
                self.assertAlmostEqual(sum(x*x for x in values[i*3:i*3+3]), 1, places=6)
        self.assertEqual(len(ids), len(set(ids)), "source rows overlap across splits")
        return manifest

    def test_synthetic_reproducibility_and_disjoint_splits(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.prepare(root / "first")
            self.prepare(root / "second")
            self.check_splits(root / "first")
            for path in (root / "first").iterdir():
                self.assertEqual(path.read_bytes(), (root / "second" / path.name).read_bytes())
            with self.assertRaises(SystemExit):
                self.prepare(root / "first")

    def test_glove_reservoir_and_source_checksum(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "glove.txt"
            source.write_text("".join(f"word{i} {i+1} 2 3\n" for i in range(40)))
            self.prepare(root / "first", "--glove", str(source))
            self.prepare(root / "second", "--glove", str(source))
            manifest = self.check_splits(root / "first")
            self.assertEqual(manifest["source"]["sha256"], hashlib.sha256(source.read_bytes()).hexdigest())
            self.assertEqual(manifest["source"]["source_rows"], 40)
            self.assertEqual((root / "first" / "manifest.json").read_bytes(),
                             (root / "second" / "manifest.json").read_bytes())

    def test_invalid_vectors_and_insufficient_data(self):
        for values in ([0, 0], [math.nan, 1], [math.inf, 1]):
            with self.assertRaises(ValueError):
                data.normalize(values)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "small.txt"
            source.write_text("one 1 2 3\n")
            with self.assertRaises(SystemExit):
                self.prepare(root / "output", "--glove", str(source))
            source.write_text("one 1 2\n")
            with self.assertRaises(SystemExit):
                self.prepare(root / "output", "--glove", str(source))

    def test_unicode_whitespace_inside_glove_tokens(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "glove.txt"
            # Matches the non-breaking spaces in tokens from the real 2024 file.
            source.write_text("".join(f".\u00a0.\u00a0.{i} {i+1} 2 3\n" for i in range(40)),
                              encoding="utf-8")
            self.prepare(root / "output", "--glove", str(source))
            self.check_splits(root / "output")


if __name__ == "__main__":
    unittest.main()
