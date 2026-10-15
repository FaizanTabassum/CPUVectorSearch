"""Reference CSV correctness, independent FP64 oracle, and provenance safety."""
import csv
import hashlib
import json
import math
from pathlib import Path
import random
import struct
import subprocess
import sys
import tempfile
import unittest

BINARY = Path(sys.argv.pop(1)).resolve()
ROOT = Path(__file__).resolve().parents[1]


def write_vectors(path, rows):
    dimensions = len(rows[0])
    payload = struct.pack(f"<{len(rows) * dimensions}f", *(x for row in rows for x in row))
    path.write_bytes(b"CVEC0001" + struct.pack("<II", len(rows), dimensions) + payload)
    rounded = struct.unpack(f"<{len(rows) * dimensions}f", payload)
    return [rounded[i:i + dimensions] for i in range(0, len(rounded), dimensions)]


class ReferenceTests(unittest.TestCase):
    def run_program(self, arguments, success=True):
        result = subprocess.run([str(BINARY), *map(str, arguments)], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0 if success else 1, result.stderr)
        if not success:
            self.assertTrue(result.stderr.startswith("error: "), result.stderr)
        return result

    def test_hand_calculated_answers_ties_and_subsets(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            write_vectors(root / "db", [[1], [2], [8], [2], [-2]])
            write_vectors(root / "queries", [[3]])
            for size, ids, distances in [(3, [1, 0, 2], [1, 4, 25]),
                                         (5, [1, 3, 0], [1, 1, 4])]:
                output = root / f"answers-{size}.csv"
                arguments = ["--database", root / "db", "--queries", root / "queries",
                             "--database-size", size, "--k", 3, "--output", output]
                self.run_program(arguments)
                with output.open() as source:
                    rows = list(csv.DictReader(source))
                self.assertEqual([int(row["neighbor_id"]) for row in rows], ids)
                self.assertEqual([float(row["squared_distance"]) for row in rows], distances)
                saved = output.read_bytes()
                self.run_program(arguments, success=False)
                self.assertEqual(output.read_bytes(), saved, "existing answers were overwritten")
            self.run_program(["--database", root / "db", "--queries", root / "queries",
                              "--database-size", 6, "--output", root / "bad"], success=False)

    def test_fp64_oracle_on_normalized_50d_inputs(self):
        rng = random.Random(5206)
        def unit():
            values = [rng.uniform(-1, 1) for _ in range(50)]
            scale = math.sqrt(sum(x * x for x in values))
            return [x / scale for x in values]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            database = write_vectors(root / "db", [unit() for _ in range(67)])
            queries = write_vectors(root / "queries", [unit() for _ in range(12)])
            for size in (10, 30, 67):
                output = root / f"n{size}.csv"
                self.run_program(["--database", root / "db", "--queries", root / "queries",
                                  "--database-size", size, "--k", 10, "--output", output])
                with output.open() as source:
                    actual = list(csv.DictReader(source))
                self.assertEqual(len(actual), len(queries) * 10)
                for q, query in enumerate(queries):
                    # FP64 arithmetic and full sorting are independent of the C++ heap.
                    expected = sorted((sum((a-b)**2 for a, b in zip(vector, query)), i)
                                      for i, vector in enumerate(database[:size]))[:10]
                    for rank, (distance, neighbor) in enumerate(expected, 1):
                        row = actual[q * 10 + rank - 1]
                        self.assertEqual((int(row["query_id"]), int(row["rank"]), int(row["neighbor_id"])),
                                         (q, rank, neighbor))
                        self.assertAlmostEqual(float(row["squared_distance"]), distance, delta=2e-5)

    def test_wrapper_reproducibility_hashes_and_heldout_is_untouched(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            folder = root / "data"
            subprocess.run([sys.executable, str(ROOT / "scripts/prepare_data.py"), "--synthetic",
                            "--database-size", "67", "--tuning-size", "3", "--test-size", "4",
                            "--output", str(folder)], check=True, capture_output=True)
            # This command must succeed without opening the held-out vector file.
            (folder / "test.cvec").unlink()
            command = [sys.executable, str(ROOT / "scripts/generate_reference.py"),
                       "--dataset", str(folder), "--binary", str(BINARY), "--sizes", "10", "30", "67"]
            for name in ("first", "second"):
                result = subprocess.run([*command, "--output", str(root / name)], capture_output=True, text=True)
                self.assertEqual(result.returncode, 0, result.stderr)
            manifest = json.loads((root / "first/reference-manifest.json").read_text())
            self.assertEqual(manifest["query_split"], "tuning")
            self.assertEqual(manifest["query_rows"], 3)
            for item in manifest["outputs"]:
                payload = (root / "first" / item["file"]).read_bytes()
                self.assertEqual(item["answer_rows"], 30)
                self.assertEqual(hashlib.sha256(payload).hexdigest(), item["sha256"])
                self.assertEqual(payload, (root / "second" / item["file"]).read_bytes())
            result = subprocess.run([*command, "--output", str(root / "first")], capture_output=True)
            self.assertNotEqual(result.returncode, 0)
            with (folder / "database.cvec").open("ab") as output:
                output.write(b"corrupt")
            result = subprocess.run([*command, "--output", str(root / "corrupt")], capture_output=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertFalse((root / "corrupt").exists())


if __name__ == "__main__":
    unittest.main()
