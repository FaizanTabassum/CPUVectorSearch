import csv
import importlib.util
from pathlib import Path
import tempfile

scripts = Path(__file__).resolve().parents[1] / "scripts"
spec = importlib.util.spec_from_file_location("plot_results", scripts / "plot_results.py")
plot_results = importlib.util.module_from_spec(spec)
spec.loader.exec_module(plot_results)

FIELDS = ["run", "route", "variant", "params", "backend", "lists", "probes", "seed",
          "database_rows", "query_rows", "split", "process_run", "recall_at_k", "recall_min",
          "latency_median_ms", "latency_p95_ms", "latency_p99_ms", "latency_mean_ms",
          "index_build_ms", "candidates_per_query", "peak_memory_bytes", "index_rss_delta_bytes",
          "rss_after_build_bytes", "peak_reset_after_load", "threads_run", "threaded_results_match",
          "qps_t1", "qps_t2", "speedup_t1", "speedup_t2"]


def row(route, variant, params, size, run, recall, median, lists="", probes="", seed="",
        backend="avx2", memory=30e6, qps=500.0):
    return {"run": f"{variant}-n{size}-r{run}", "route": route, "variant": variant,
            "params": params, "backend": backend, "lists": lists, "probes": probes, "seed": seed,
            "database_rows": size, "query_rows": 500, "split": "tuning", "process_run": run,
            "recall_at_k": recall, "recall_min": recall, "latency_median_ms": median,
            "latency_p95_ms": median * 1.5, "latency_p99_ms": median * 2,
            "latency_mean_ms": median * 1.1, "index_build_ms": 5.0,
            "candidates_per_query": size if route != "ivf" else size / 4,
            "peak_memory_bytes": memory, "index_rss_delta_bytes": 1e6,
            "rss_after_build_bytes": memory * 0.9, "peak_reset_after_load": "False",
            "threads_run": "1 2", "threaded_results_match": "True",
            "qps_t1": qps, "qps_t2": qps * 1.8, "speedup_t1": 1.0, "speedup_t2": 1.8}


rows = []
for size in (10000, 100000):
    scale = size / 10000
    for run, jitter in ((1, 1.0), (2, 1.2), (3, 0.9)):
        rows.append(row("scalar_flat", "scalar_flat", "", size, run, 1.0, 0.22 * scale * jitter,
                        backend="scalar"))
        rows.append(row("optimized_flat", "optimized_flat", "", size, run, 1.0,
                        0.12 * scale * jitter, memory=45e6))
        for probes, base_recall, base_time in ((8, 0.88, 0.035), (16, 0.95, 0.07)):
            for seed, shift in ((5206, -0.01), (5207, 0.0), (5208, 0.01)):
                params = f"lists=64;probes={probes};seed={seed}"
                rows.append(row("ivf", f"ivf-l64-p{probes}-s{seed}", params, size, run,
                                base_recall + shift, base_time * scale * jitter,
                                lists=64, probes=probes, seed=seed, memory=26e6))

with tempfile.TemporaryDirectory() as temporary:
    run_dir = Path(temporary) / "run"
    run_dir.mkdir()
    with (run_dir / "summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    plot_results.main([str(run_dir), "--no-plots"])
    report = run_dir / "report"
    summary = (report / "summary.md").read_text()
    aggregated = list(csv.DictReader((report / "aggregated.csv").open()))
    assert len(aggregated) == 2 * 4, len(aggregated)
    assert "ivf (lists=64;probes=16)" in summary, summary
    assert "ivf (lists=64;probes=8)" not in summary.split("## IVF sweep")[0]
    assert "0.9400-0.9600" in summary
    small = [r for r in aggregated if r["variant"] == "optimized_flat" and r["database_rows"] == "10000"]
    assert abs(float(small[0]["latency_median_ms"]) - 0.12) < 1e-9
    ivf16 = [r for r in aggregated if "probes=16" in r["params"] and r["database_rows"] == "100000"]
    assert abs(float(ivf16[0]["recall_at_k"]) - 0.95) < 1e-9 and ivf16[0]["seeds"] == "3"

    try:
        import matplotlib
    except ImportError:
        matplotlib = None
    if matplotlib is not None:
        figures = Path(temporary) / "figures"
        plot_results.main([str(run_dir), "--output", str(figures)])
        for name in ["latency_by_size.png", "recall_vs_latency.png", "thread_scaling.png",
                     "memory_by_size.png", "build_time.png"]:
            assert (figures / name).stat().st_size > 1000, name
    else:
        print("matplotlib not installed; figures not checked")

print("Plot and summary checks passed.")
