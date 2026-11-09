#!/usr/bin/env python3
import argparse
import csv
import datetime
import hashlib
import itertools
import json
from pathlib import Path
import platform
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
DESCRIPTION = "Run the vector_search benchmark matrix and record everything needed to reproduce it."


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def cvec_shape(path):
    with Path(path).open("rb") as source:
        header = source.read(16)
    if len(header) != 16 or header[:8] != b"CVEC0001":
        raise ValueError(f"{path} is not a CVEC0001 file")
    rows = int.from_bytes(header[8:12], "little")
    dims = int.from_bytes(header[12:16], "little")
    return rows, dims


def checked_split(folder, manifest, name):
    """Return the split path after checking it matches the dataset manifest."""
    split = manifest["splits"][name]
    filename = split["file"]
    if Path(filename).name != filename:
        raise ValueError("split files must sit directly inside the dataset directory")
    path = folder / filename
    digest = sha256(path)
    if digest != split["sha256"]:
        raise ValueError(f"{name} file does not match its manifest checksum")
    rows, dims = cvec_shape(path)
    if rows != split["rows"] or dims != manifest["dimensions"]:
        raise ValueError(f"{name} shape does not match the manifest")
    return path, digest, rows


def git_state():
    def git(*args):
        result = subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True)
        return result.stdout.strip() if result.returncode == 0 else None
    commit = git("rev-parse", "HEAD")
    status = git("status", "--porcelain")
    return {"commit": commit, "dirty": bool(status) if status is not None else None}


def build_settings(binary):
    """Compiler and CMake settings, read from the cache next to the binary."""
    cache = Path(binary).parent / "CMakeCache.txt"
    if not cache.is_file():
        return None
    wanted = {"CMAKE_CXX_COMPILER", "CMAKE_BUILD_TYPE", "CMAKE_CXX_FLAGS",
              "CMAKE_CXX_FLAGS_RELEASE", "CVS_SANITIZERS"}
    settings = {}
    for line in cache.read_text().splitlines():
        if "=" in line and ":" in line.split("=", 1)[0]:
            key, value = line.split("=", 1)
            key = key.split(":", 1)[0]
            if key in wanted:
                settings[key] = value
    compiler = settings.get("CMAKE_CXX_COMPILER")
    if compiler:
        result = subprocess.run([compiler, "--version"], capture_output=True, text=True)
        if result.returncode == 0:
            settings["compiler_version"] = result.stdout.strip().splitlines()[0]
    return settings


def find_program(path):
    path = Path(path)
    if path.is_file():
        return path.resolve()
    windows = path.with_name(path.name + ".exe")
    if windows.is_file():
        return windows.resolve()
    raise ValueError(f"{path.name} not found; build the project first")


def read_freeze(path):
    if not path.is_file():
        raise ValueError(f"freeze record not found: {path}")
    record = json.loads(path.read_text())
    if record.get("status") != "frozen":
        raise ValueError("the evaluation freeze record is not frozen yet; the test split stays "
                         "locked until A commits the frozen settings. Tune on --split tuning.")
    return {"file": str(path), "sha256": sha256(path), "frozen_at_utc": record.get("frozen_at_utc"),
            "frozen_code_commit": record.get("frozen_code_commit")}


def available_routes(binary):
    result = subprocess.run([str(binary), "--list-routes"], capture_output=True, text=True,
                            check=True)
    return result.stdout.split()


def route_variants(route, config, args):
    """Expand one route name into a list of (label, params dict)."""
    if route == "scalar_flat":
        return [("scalar_flat", {})]
    if route == "optimized_flat":
        out = []
        for backend in args.optimized_backends:
            params = {} if backend == "auto" else {"backend": backend}
            label = "optimized_flat" if backend == "auto" else f"optimized_flat-{backend}"
            out.append((label, params))
        return out
    if route == "ivf":
        search = config["search"]
        lists_values = args.ivf_lists or search["ivf_lists"]
        seeds = args.ivf_seeds or search["training_seeds"]
        out = []
        for lists in lists_values:
            probes_values = args.ivf_probes or search["ivf_probe_counts"].get(str(lists), [])
            for probes, seed in itertools.product(probes_values, seeds):
                if probes > lists:
                    continue
                params = {"lists": lists, "probes": probes, "seed": seed}
                if args.ivf_backend != "auto":
                    params["backend"] = args.ivf_backend
                out.append((f"ivf-l{lists}-p{probes}-s{seed}", params))
        return out
    return [(route, {})]


SUMMARY_FIELDS = [
    "run", "route", "variant", "params", "backend", "lists", "probes", "seed",
    "database_rows", "query_rows", "split", "process_run", "recall_at_k", "recall_min",
    "latency_median_ms", "latency_p95_ms", "latency_p99_ms", "latency_mean_ms",
    "index_build_ms", "candidates_per_query", "peak_memory_bytes", "index_rss_delta_bytes",
    "rss_after_build_bytes", "peak_reset_after_load", "threads_run", "threaded_results_match",
]


def summary_row(run_name, variant, params, split, process_run, result):
    row = {
        "run": run_name, "route": result["route"], "variant": variant,
        "params": ";".join(f"{k}={v}" for k, v in sorted(params.items())),
        "backend": result["route_metadata"].get("backend", ""),
        "lists": params.get("lists", ""), "probes": params.get("probes", ""),
        "seed": params.get("seed", ""), "database_rows": result["database_rows"],
        "query_rows": result["query_rows"], "split": split, "process_run": process_run,
        "threads_run": " ".join(str(t) for t in result["threads_run"]),
    }
    for key in ["recall_at_k", "recall_min", "latency_median_ms", "latency_p95_ms",
                "latency_p99_ms", "latency_mean_ms", "index_build_ms",
                "candidates_per_query", "peak_memory_bytes", "threaded_results_match"]:
        row[key] = result[key]
    for key in ["index_rss_delta_bytes", "rss_after_build_bytes", "peak_reset_after_load"]:
        row[key] = result["memory"][key]
    for batch in result["batch"]:
        t = batch["threads"]
        row[f"qps_t{t}"] = batch["queries_per_second"]
        row[f"speedup_t{t}"] = batch["speedup_vs_1_thread"]
    return row


def main(argv=None):
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    parser.add_argument("--dataset", type=Path, default=ROOT / "data/processed/glove-50d-seed5206",
                        help="prepared dataset directory containing manifest.json")
    parser.add_argument("--binary", type=Path, default=ROOT / "build/vector_search")
    parser.add_argument("--split", choices=["tuning", "test"], default="tuning")
    parser.add_argument("--output", type=Path, required=True, help="new directory for results")
    parser.add_argument("--config", type=Path, default=ROOT / "configs/experiments.json")
    parser.add_argument("--reference-dir", type=Path,
                        help="answer sheets; default DATASET/reference-SPLIT")
    parser.add_argument("--generate-references", action="store_true",
                        help="development only: make answer sheets with reference_search "
                             "inside the output directory instead of using Member A's")
    parser.add_argument("--reference-binary", type=Path, default=ROOT / "build/reference_search")
    parser.add_argument("--freeze-record", type=Path, default=ROOT / "configs/evaluation-freeze.json",
                        help="A's freeze record; must say frozen before --split test runs")
    parser.add_argument("--sizes", type=int, nargs="+")
    parser.add_argument("--routes", nargs="+")
    parser.add_argument("--threads", type=int, nargs="+", help="batch thread counts")
    parser.add_argument("--optimized-backends", nargs="+", default=["auto"],
                        choices=["auto", "scalar", "avx2"])
    parser.add_argument("--ivf-lists", type=int, nargs="+")
    parser.add_argument("--ivf-probes", type=int, nargs="+")
    parser.add_argument("--ivf-seeds", type=int, nargs="+")
    parser.add_argument("--ivf-backend", default="auto", choices=["auto", "scalar", "avx2"])
    parser.add_argument("--k", type=int)
    parser.add_argument("--warmup", type=int)
    parser.add_argument("--repetitions", type=int)
    parser.add_argument("--batch-repetitions", type=int)
    parser.add_argument("--process-runs", type=int,
                        help="separate processes per configuration (medians taken when plotting)")
    parser.add_argument("--strict", action="store_true",
                        help="fail instead of skipping routes this build does not provide")
    parser.add_argument("--dry-run", action="store_true", help="print commands only")
    args = parser.parse_args(argv)

    try:
        config = json.loads(args.config.read_text())
        evaluation, search = config["evaluation"], config["search"]
        k = args.k or search["k"]
        sizes = args.sizes or config["dataset"]["database_sizes"]
        routes = args.routes or search["routes"]
        threads = args.threads or evaluation["batch_threads"]
        warmup = args.warmup if args.warmup is not None else (evaluation.get("warmup_queries") or 10)
        repetitions = args.repetitions or evaluation.get("repetitions") or 5
        args.batch_repetitions = args.batch_repetitions or evaluation.get("batch_repetitions") or 3
        args.process_runs = args.process_runs or evaluation.get("process_runs") or 1
        if min([k, warmup + 1, repetitions, args.batch_repetitions, args.process_runs] + threads) <= 0:
            raise ValueError("k, repetitions, process runs and thread counts must be positive")

        freeze = read_freeze(args.freeze_record) if args.split == "test" else None

        dataset = args.dataset.resolve()
        manifest_path = dataset / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("format") != "CVEC0001":
            raise ValueError("unsupported dataset format")
        database, database_sha, database_rows = checked_split(dataset, manifest, "database")
        queries, queries_sha, query_rows = checked_split(dataset, manifest, args.split)
        if len(set(sizes)) != len(sizes) or any(n < k or n > database_rows for n in sizes):
            raise ValueError(f"sizes must be unique and between k and {database_rows} "
                             "(use --sizes for small development datasets)")

        binary = find_program(args.binary)
        built = available_routes(binary)
        runnable = []
        for route in routes:
            if route in built:
                runnable.append(route)
            elif args.strict:
                raise ValueError(f"route {route} is not in this build")
            else:
                print(f"note: skipping {route}; this build does not provide it yet",
                      file=sys.stderr)
        if not runnable:
            raise ValueError("no requested route is available in this build")

        output = args.output.resolve()
        if output.exists():
            raise ValueError("output directory already exists; choose a new one")

        references = {}
        reference_source = "member_a"
        if args.generate_references:
            if args.split == "test":
                raise ValueError("generate test answers with Member A's process after the freeze")
            reference_source = "generated_by_run_benchmarks_for_development"
            reference_dir = output / f"reference-{args.split}"
        else:
            reference_dir = (args.reference_dir or dataset / f"reference-{args.split}").resolve()
            ref_manifest_path = reference_dir / "reference-manifest.json"
            ref_manifest = (json.loads(ref_manifest_path.read_text())
                            if ref_manifest_path.is_file() else None)
            if ref_manifest:
                if ref_manifest.get("database_sha256") not in (None, database_sha):
                    raise ValueError("answer sheets were made from a different database file")
                if ref_manifest.get("queries_sha256") not in (None, queries_sha):
                    raise ValueError("answer sheets were made from a different query file")
            for size in sizes:
                path = reference_dir / f"top{k}-{args.split}-n{size}.csv"
                if not path.is_file():
                    raise ValueError(f"missing answer sheet {path}; ask Member A, or use "
                                     "--generate-references while developing")
                digest = sha256(path)
                if ref_manifest:
                    listed = {o["file"]: o for o in ref_manifest.get("outputs", [])}
                    entry = listed.get(path.name)
                    if entry is None or entry["sha256"] != digest or entry["database_rows"] != size:
                        raise ValueError(f"{path.name} does not match reference-manifest.json")
                references[size] = {"file": str(path), "sha256": digest}

        git = git_state()
        common_metadata = {
            "split": args.split, "data_seed": manifest.get("seed"),
            "database_sha256": database_sha, "queries_sha256": queries_sha,
            "git_commit": git["commit"], "git_dirty": git["dirty"],
        }
        if freeze:
            common_metadata["freeze_record_sha256"] = freeze["sha256"]

        jobs = []
        for process_run in range(1, args.process_runs + 1):
            for size in sizes:
                for route in runnable:
                    for variant, params in route_variants(route, config, args):
                        name = f"{variant}-n{size}-r{process_run}"
                        jobs.append((name, route, variant, params, size, process_run))

        def command_for(name, route, params, size, process_run, reference_file):
            cmd = [str(binary), "--database", str(database), "--queries", str(queries),
                   "--database-size", str(size), "--route", route, "--k", str(k),
                   "--ground-truth", reference_file,
                   "--warmup-queries", str(warmup), "--repetitions", str(repetitions),
                   "--threads", ",".join(map(str, threads)),
                   "--batch-repetitions", str(args.batch_repetitions),
                   "--label", name,
                   "--output", str(output / "runs" / f"{name}.json"),
                   "--raw-output", str(output / "runs" / f"{name}-queries.csv")]
            for key, value in sorted(params.items()):
                cmd += ["--param", f"{key}={value}"]
            metadata = dict(common_metadata, process_run=process_run,
                            reference_sha256=references.get(size, {}).get("sha256", "pending"))
            for key, value in sorted(metadata.items()):
                cmd += ["--metadata", f"{key}={value}"]
            return cmd

        if args.dry_run:
            for name, route, _, params, size, run in jobs:
                ref = references.get(size, {}).get("file", str(reference_dir / f"top{k}-{args.split}-n{size}.csv"))
                print(shlex.join(command_for(name, route, params, size, run, ref)))
            return 0

        (output / "runs").mkdir(parents=True)
        if args.generate_references:
            reference_dir.mkdir()
            for size in sizes:
                path = reference_dir / f"top{k}-{args.split}-n{size}.csv"
                subprocess.run([str(find_program(args.reference_binary)), "--database", str(database),
                                "--queries", str(queries), "--database-size", str(size),
                                "--k", str(k), "--output", str(path)], check=True,
                               capture_output=True)
                references[size] = {"file": str(path), "sha256": sha256(path)}

        rows, commands = [], []
        for index, (name, route, variant, params, size, process_run) in enumerate(jobs, 1):
            cmd = command_for(name, route, params, size, process_run, references[size]["file"])
            commands.append(shlex.join(cmd))
            print(f"[{index}/{len(jobs)}] {name}", flush=True)
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                raise RuntimeError(f"{name} failed:\n{result.stderr}")
            if result.stderr.strip():
                print(result.stderr.strip(), file=sys.stderr)
            summary = json.loads((output / "runs" / f"{name}.json").read_text())
            rows.append(summary_row(name, variant, params, args.split, process_run, summary))

        thread_fields = sorted({key for row in rows for key in row if key.startswith(("qps_t", "speedup_t"))},
                               key=lambda key: (key.split("_t")[0], int(key.split("_t")[1])))
        with (output / "summary.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=SUMMARY_FIELDS + thread_fields)
            writer.writeheader()
            writer.writerows(rows)

        first = json.loads((output / "runs" / f"{jobs[0][0]}.json").read_text())
        run_manifest = {
            "schema_version": 1,
            "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
            "purpose": "final_test_evaluation" if args.split == "test" else "tuning_or_development",
            "split": args.split, "k": k, "sizes": sizes, "routes": runnable,
            "skipped_routes": [r for r in routes if r not in runnable],
            "threads_requested": threads, "warmup_queries": warmup, "repetitions": repetitions,
            "batch_repetitions": args.batch_repetitions, "process_runs": args.process_runs,
            "git": git, "binary": str(binary), "binary_sha256": sha256(binary),
            "build_settings": build_settings(binary),
            "environment": first["environment"], "hardware_threads": first["hardware_threads"],
            "host_platform": platform.platform(), "python": platform.python_version(),
            "memory_method": first["memory"]["method"],
            "dataset": {"directory": str(dataset), "manifest_sha256": sha256(manifest_path),
                        "data_seed": manifest.get("seed"), "dimensions": manifest["dimensions"],
                        "database_rows": database_rows, "database_sha256": database_sha,
                        "query_rows": query_rows, "queries_sha256": queries_sha},
            "references": {"source": reference_source, "directory": str(reference_dir),
                           "files": {str(size): ref for size, ref in references.items()}},
            "freeze_record": freeze,
            "config_file": str(args.config), "config_sha256": sha256(args.config),
            "commands": commands,
        }
        (output / "run-manifest.json").write_text(json.dumps(run_manifest, indent=2) + "\n")
        print(f"Saved {len(rows)} runs in {output}")
        return 0
    except (OSError, ValueError, KeyError, RuntimeError, subprocess.CalledProcessError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    sys.exit(main())
