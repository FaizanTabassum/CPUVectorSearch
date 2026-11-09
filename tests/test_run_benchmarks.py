import csv
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile

benchmark = Path(sys.argv[1]).resolve()
reference = Path(sys.argv[2]).resolve()
root_dir = Path(__file__).resolve().parents[1]
scripts = root_dir / "scripts"


def load(name):
    spec = importlib.util.spec_from_file_location(name, scripts / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def driver(*args, success=True):
    result = subprocess.run([sys.executable, str(scripts / "run_benchmarks.py"),
                             "--binary", str(benchmark), "--reference-binary", str(reference),
                             *map(str, args)], capture_output=True, text=True)
    if (result.returncode == 0) != success:
        raise AssertionError(f"driver {args} returned {result.returncode}:\n{result.stdout}\n{result.stderr}")
    return result


def read_rows(path):
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


run_benchmarks = load("run_benchmarks")

config = json.loads((root_dir / "configs/experiments.json").read_text())


class Args:
    optimized_backends = ["auto", "scalar"]
    ivf_lists = ivf_probes = ivf_seeds = None
    ivf_backend = "auto"


ivf = run_benchmarks.route_variants("ivf", config, Args)
assert len(ivf) == (7 + 9) * 3, len(ivf)
assert all(v[1]["probes"] <= v[1]["lists"] for v in ivf)
assert ("ivf-l64-p8-s5207", {"lists": 64, "probes": 8, "seed": 5207}) in ivf
Args.ivf_backend = "scalar"
assert run_benchmarks.route_variants("ivf", config, Args)[0][1]["backend"] == "scalar"
assert [v[0] for v in run_benchmarks.route_variants("optimized_flat", config, Args)] == [
    "optimized_flat", "optimized_flat-scalar"]

with tempfile.TemporaryDirectory() as temporary:
    root = Path(temporary)
    data = root / "data"
    subprocess.run([sys.executable, str(scripts / "prepare_data.py"), "--synthetic",
                    "--database-size", "400", "--tuning-size", "10", "--test-size", "6",
                    "--seed", "3", "--output", str(data)], check=True, capture_output=True)

    out = root / "run"
    driver("--dataset", data, "--output", out, "--generate-references", "--sizes", 100, 400,
           "--routes", "scalar_flat", "optimized_flat", "--repetitions", 2,
           "--batch-repetitions", 1, "--process-runs", 2, "--threads", 1, 2)
    rows = read_rows(out / "summary.csv")
    assert len(rows) == 8, len(rows)
    assert all(float(r["recall_at_k"]) == 1.0 for r in rows)
    assert {r["database_rows"] for r in rows} == {"100", "400"}
    assert all(r["threaded_results_match"] == "True" for r in rows)
    manifest = json.loads((out / "run-manifest.json").read_text())
    assert manifest["split"] == "tuning" and manifest["freeze_record"] is None
    assert manifest["references"]["source"].startswith("generated")
    assert len(manifest["commands"]) == 8 and manifest["dataset"]["database_rows"] == 400
    one = json.loads((out / "runs" / "scalar_flat-n100-r1.json").read_text())
    assert one["metadata"]["split"] == "tuning" and one["metadata"]["process_run"] == "1"
    assert one["metadata"]["database_sha256"] == manifest["dataset"]["database_sha256"]

    sweep = root / "sweep"
    driver("--dataset", data, "--output", sweep, "--generate-references", "--sizes", 400,
           "--routes", "ivf", "--ivf-lists", 4, "--ivf-probes", 1, 4, "--ivf-seeds", 1, 2,
           "--ivf-backend", "scalar", "--repetitions", 1, "--batch-repetitions", 1,
           "--process-runs", 1, "--threads", 1)
    rows = read_rows(sweep / "summary.csv")
    assert sorted(r["variant"] for r in rows) == [
        "ivf-l4-p1-s1", "ivf-l4-p1-s2", "ivf-l4-p4-s1", "ivf-l4-p4-s2"], rows
    full = [r for r in rows if r["probes"] == "4"]
    assert all(float(r["recall_at_k"]) == 1.0 and r["backend"] == "scalar" for r in full)
    assert all(float(r["candidates_per_query"]) < 400 for r in rows if r["probes"] == "1")

    driver("--dataset", data, "--output", out, "--generate-references", "--sizes", 100,
           success=False)
    driver("--dataset", data, "--output", root / "big", "--generate-references", "--sizes", 500,
           success=False)
    driver("--dataset", data, "--output", root / "noref", "--sizes", 100, success=False)

    pending = root / "pending.json"
    pending.write_text('{"status": "pending"}\n')
    driver("--dataset", data, "--output", root / "t1", "--split", "test", "--sizes", 100,
           "--freeze-record", pending, success=False)
    frozen = root / "frozen.json"
    frozen.write_text('{"status": "frozen"}\n')
    driver("--dataset", data, "--output", root / "t2", "--split", "test", "--sizes", 100,
           "--freeze-record", frozen, "--generate-references", success=False)

    ref_dir = root / "refs"
    ref_dir.mkdir()
    sheet = ref_dir / "top10-tuning-n100.csv"
    sheet.write_bytes((out / "reference-tuning" / "top10-tuning-n100.csv").read_bytes())
    (ref_dir / "reference-manifest.json").write_text(json.dumps({
        "database_sha256": manifest["dataset"]["database_sha256"],
        "queries_sha256": manifest["dataset"]["queries_sha256"],
        "outputs": [{"file": sheet.name, "database_rows": 100,
                     "sha256": run_benchmarks.sha256(sheet)}]}))
    driver("--dataset", data, "--output", root / "ok", "--reference-dir", ref_dir,
           "--sizes", 100, "--routes", "scalar_flat", "--repetitions", 1, "--threads", 1,
           "--process-runs", 1)
    sheet.write_text(sheet.read_text().replace("0,1,", "0,1,9", 1))
    driver("--dataset", data, "--output", root / "tampered", "--reference-dir", ref_dir,
           "--sizes", 100, "--routes", "scalar_flat", success=False)

    (data / "tuning.cvec").write_bytes((data / "tuning.cvec").read_bytes()[:-4] + b"\0\0\0\0")
    driver("--dataset", data, "--output", root / "corrupt", "--generate-references",
           "--sizes", 100, success=False)

print("Experiment driver checks passed.")
