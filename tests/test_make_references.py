import csv
import json
from pathlib import Path
import subprocess
import sys
import tempfile

benchmark = Path(sys.argv[1]).resolve()
reference = Path(sys.argv[2]).resolve()
scripts = Path(__file__).resolve().parents[1] / "scripts"


def make(*args, success=True):
    result = subprocess.run([sys.executable, str(scripts / "make_references.py"),
                             "--binary", str(benchmark), *map(str, args)],
                            capture_output=True, text=True)
    if (result.returncode == 0) != success:
        raise AssertionError(f"make_references {args} returned {result.returncode}:\n"
                             f"{result.stdout}\n{result.stderr}")
    return result


with tempfile.TemporaryDirectory() as temporary:
    root = Path(temporary)
    data = root / "data"
    subprocess.run([sys.executable, str(scripts / "prepare_data.py"), "--synthetic",
                    "--database-size", "300", "--tuning-size", "12", "--test-size", "4",
                    "--seed", "9", "--output", str(data)], check=True, capture_output=True)

    scalar = root / "scalar"
    make("--dataset", data, "--sizes", 100, 300, "--backend", "scalar", "--output", scalar)
    manifest = json.loads((scalar / "reference-manifest.json").read_text())
    assert manifest["backend"] == "scalar" and manifest["split"] == "tuning"
    assert [o["database_rows"] for o in manifest["outputs"]] == [100, 300]
    for size in (100, 300):
        exact = root / f"exact-{size}.csv"
        subprocess.run([str(reference), "--database", str(data / "database.cvec"),
                        "--queries", str(data / "tuning.cvec"), "--database-size", str(size),
                        "--output", str(exact)], check=True, capture_output=True)
        made = scalar / f"top10-tuning-n{size}.csv"
        assert made.read_text() == exact.read_text(), size

    fast = root / "fast"
    make("--dataset", data, "--sizes", 100, "--backend", "auto", "--output", fast)
    manifest = json.loads((fast / "reference-manifest.json").read_text())
    assert manifest["backend"] in ("avx2", "scalar")
    with (fast / "top10-tuning-n100.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 12 * 10 and rows[0]["rank"] == "1"

    run = root / "run"
    subprocess.run([sys.executable, str(scripts / "run_benchmarks.py"), "--binary", str(benchmark),
                    "--dataset", str(data), "--reference-dir", str(fast), "--sizes", "100",
                    "--routes", "optimized_flat", "--repetitions", "1", "--batch-repetitions", "1",
                    "--process-runs", "1", "--threads", "1", "--output", str(run)],
                   check=True, capture_output=True)
    with (run / "summary.csv").open(newline="") as handle:
        assert float(next(csv.DictReader(handle))["recall_at_k"]) == 1.0

    make("--dataset", data, "--sizes", 100, "--backend", "scalar", "--output", scalar,
         success=False)
    make("--dataset", data, "--sizes", 301, "--output", root / "big", success=False)
    assert not (root / "big").exists()
    pending = root / "pending.json"
    pending.write_text('{"status": "pending"}\n')
    make("--dataset", data, "--split", "test", "--sizes", 100, "--freeze-record", pending,
         "--output", root / "test", success=False)
    frozen = root / "frozen.json"
    frozen.write_text('{"status": "frozen"}\n')
    make("--dataset", data, "--split", "test", "--sizes", 100, "--freeze-record", frozen,
         "--backend", "scalar", "--output", root / "test")
    assert json.loads((root / "test" / "reference-manifest.json").read_text())["freeze_record"]

print("Backend-matched answer sheet checks passed.")
