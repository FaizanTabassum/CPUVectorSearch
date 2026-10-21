
import csv
import json
from pathlib import Path
import subprocess
import sys
import tempfile

benchmark = Path(sys.argv[1]).resolve()
reference = Path(sys.argv[2]).resolve()
prepare = Path(__file__).resolve().parents[1] / "scripts" / "prepare_data.py"


def run(*args, success=True):
    result = subprocess.run([str(benchmark), *map(str, args)], capture_output=True, text=True)
    ok = result.returncode == 0 if success else (
        result.returncode == 1 and result.stderr.startswith("error: "))
    if not ok:
        raise AssertionError(f"unexpected result for {args}:\n{result.stdout}\n{result.stderr}")
    return result


def read_csv(path):
    with open(path, newline="") as handle:
        return list(csv.DictReader(handle))


with tempfile.TemporaryDirectory() as temporary:
    root = Path(temporary)
    data = root / "data"
    subprocess.run([sys.executable, str(prepare), "--synthetic", "--database-size", "300",
                    "--tuning-size", "12", "--test-size", "3", "--seed", "11",
                    "--output", str(data)], check=True, capture_output=True)
    database, queries = data / "database.cvec", data / "tuning.cvec"

    
    truth = root / "truth-n200.csv"
    subprocess.run([str(reference), "--database", str(database), "--queries", str(queries),
                    "--database-size", "200", "--output", str(truth)], check=True,
                   capture_output=True)

    
    out, raw = root / "s.json", root / "raw.csv"
    run("--database", database, "--queries", queries, "--database-size", 200,
        "--ground-truth", truth, "--repetitions", 2, "--output", out,
        "--raw-output", raw, "--metadata", "split=tuning")
    summary = json.loads(out.read_text())
    assert summary["schema_version"] == 2 and summary["route"] == "scalar_flat"
    assert summary["database_rows"] == 200 and summary["database_file_rows"] == 300
    assert summary["candidates_per_query"] == 200
    assert summary["recall_at_k"] == 1.0 and summary["recall_min"] == 1.0
    assert summary["recall_perfect_queries"] == 12
    assert summary["batch_queries_per_second"] > 0
    assert summary["latency_p99_ms"] >= summary["latency_p95_ms"] >= summary["latency_median_ms"]
    assert summary["metadata"] == {"split": "tuning"}
    rows = read_csv(raw)
    assert len(rows) == 24 and {r["repetition"] for r in rows} == {"0", "1"}
    assert all(float(r["recall_at_k"]) == 1.0 for r in rows)
    assert all(int(r["candidates_examined"]) == 200 for r in rows)

   
    rows = read_csv(truth)
    by_query = {}
    for row in rows:
        by_query.setdefault(row["query_id"], set()).add(int(row["neighbor_id"]))
    for row in rows:
        if row["rank"] in ("1", "2"):
            unused = sorted(set(range(200)) - by_query[row["query_id"]])
            row["neighbor_id"] = str(unused[int(row["rank"]) - 1])
    broken = root / "broken.csv"
    fields = ["query_id", "rank", "neighbor_id", "squared_distance"]
    with broken.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    out, raw = root / "broken.json", root / "broken-raw.csv"
    run("--database", database, "--queries", queries, "--database-size", 200,
        "--ground-truth", broken, "--output", out, "--raw-output", raw)
    summary = json.loads(out.read_text())
    assert abs(summary["recall_at_k"] - 0.8) < 1e-12, summary["recall_at_k"]
    assert summary["recall_perfect_queries"] == 0
    assert all(abs(float(r["recall_at_k"]) - 0.8) < 1e-9 for r in read_csv(raw))

    
    rows[0]["neighbor_id"] = "250"
    with broken.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    run("--database", database, "--queries", queries, "--database-size", 200,
        "--ground-truth", broken, success=False)

   
    out = root / "k5.json"
    run("--database", database, "--queries", queries, "--database-size", 200,
        "--ground-truth", truth, "--k", 5, "--output", out)
    assert json.loads(out.read_text())["recall_at_k"] == 1.0

   
    out = root / "plain.json"
    run("--database", database, "--queries", queries, "--output", out)
    plain = json.loads(out.read_text())
    assert plain["recall_at_k"] is None and plain["ground_truth_file"] is None

    
    other_queries = data / "test.cvec"
    bad_runs = [
        ["--ground-truth", truth, "--k", 11],                      
        ["--ground-truth", truth, "--queries-dummy", "x"],         
        ["--database-size", 301],                                   
        ["--database-size", 0],
        ["--repetitions", 0],
        ["--metadata", "novalue"],
    ]
    for extra in bad_runs:
        run("--database", database, "--queries", queries, *extra, success=False)
    
    run("--database", database, "--queries", other_queries, "--ground-truth", truth, success=False)

print("Benchmark CLI checks passed: recall, raw outputs and input validation.")
