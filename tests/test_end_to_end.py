"""Cross-language IO and benchmark CLI integration; no timing thresholds."""
import json
from pathlib import Path
import struct
import subprocess
import sys
import tempfile

binary = Path(sys.argv[1]).resolve()
prepare = Path(__file__).resolve().parents[1] / "scripts" / "prepare_data.py"


def run(*args, success=True):
    result = subprocess.run([str(binary), *map(str, args)], capture_output=True, text=True)
    valid = result.returncode == 0 if success else (
        result.returncode == 1 and result.stderr.startswith("error: "))
    if not valid:
        raise RuntimeError(f"unexpected CLI result: {result.stdout}\n{result.stderr}")
    return result


with tempfile.TemporaryDirectory() as temporary:
    root = Path(temporary)
    subprocess.run([sys.executable, str(prepare), "--synthetic", "--database-size", "40",
                    "--tuning-size", "5", "--test-size", "7", "--output", str(root / "data")], check=True)
    database, queries = root / "data/database.cvec", root / "data/tuning.cvec"
    output = root / "result.json"
    run("--database", database, "--queries", queries, "--output", output)
    result = json.loads(output.read_text())
    assert result["route"] == "scalar_flat"
    assert result["database_rows"] == 40 and result["query_rows"] == 5
    assert result["candidates_per_query"] == 40 and result["database_vector_bytes"] == 40 * 50 * 4
    assert result["latency_p95_ms"] >= result["latency_median_ms"] >= 0
    assert result["batch_queries_per_second"] > 0 and result["recall_at_k"] is None
    for bad_k in ("0", "-1", "1x", "41"):
        run("--database", database, "--queries", queries, "--k", bad_k, success=False)
    run("--unknown", "value", success=False)
    run("--database", root / "missing", "--queries", queries, success=False)
    valid = database.read_bytes()
    corruptions = [b"bad", valid[:-1], valid + b"extra",
                   b"CVEC0001" + struct.pack("<II", 0xFFFFFFFF, 0xFFFFFFFF),
                   b"CVEC0001" + struct.pack("<II", 1, 0),
                   b"CVEC0001" + struct.pack("<II", 1, 50) + struct.pack("<50f", *([float("nan")] * 50))]
    for payload in corruptions:
        bad = root / "bad.cvec"
        bad.write_bytes(payload)
        run("--database", bad, "--queries", queries, success=False)
    # Dimension mismatch must be caught before search.
    mismatch = root / "mismatch.cvec"
    mismatch.write_bytes(b"CVEC0001" + struct.pack("<II2f", 1, 2, 1, 0))
    run("--database", database, "--queries", mismatch, success=False)
print("End-to-end data preparation, CVEC loading, CLI validation and JSON checks passed.")
