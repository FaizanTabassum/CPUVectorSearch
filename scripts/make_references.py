#!/usr/bin/env python3
import argparse
import datetime
import json
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import tempfile

from run_benchmarks import (ROOT, checked_split, find_program, git_state, read_freeze,
                            sha256)

DESCRIPTION = ("Make exhaustive top-k answer sheets with a chosen scoring backend, "
               "in the same format and manifest layout as A's reference answers.")


def main(argv=None):
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    parser.add_argument("--dataset", type=Path, default=ROOT / "data/processed/glove-50d-seed5206")
    parser.add_argument("--binary", type=Path, default=ROOT / "build/vector_search")
    parser.add_argument("--split", choices=["tuning", "test"], default="tuning")
    parser.add_argument("--backend", choices=["avx2", "scalar", "auto"], default="avx2")
    parser.add_argument("--sizes", type=int, nargs="+")
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--output", type=Path,
                        help="new directory; default DATASET/reference-SPLIT-BACKEND")
    parser.add_argument("--config", type=Path, default=ROOT / "configs/experiments.json")
    parser.add_argument("--freeze-record", type=Path, default=ROOT / "configs/evaluation-freeze.json")
    args = parser.parse_args(argv)

    try:
        config = json.loads(args.config.read_text())
        sizes = args.sizes or config["dataset"]["database_sizes"]
        if args.k <= 0:
            raise ValueError("k must be positive")
        freeze = read_freeze(args.freeze_record) if args.split == "test" else None

        dataset = args.dataset.resolve()
        manifest_path = dataset / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        database, database_sha, database_rows = checked_split(dataset, manifest, "database")
        queries, queries_sha, query_rows = checked_split(dataset, manifest, args.split)
        if len(set(sizes)) != len(sizes) or any(n < args.k or n > database_rows for n in sizes):
            raise ValueError(f"sizes must be unique and between k and {database_rows}")

        binary = find_program(args.binary)
        output = (args.output or dataset / f"reference-{args.split}-{args.backend}").resolve()
        if output.exists():
            raise ValueError(f"{output} already exists; choose a new folder")
        output.mkdir(parents=True)
    except (OSError, ValueError, KeyError) as error:
        parser.error(str(error))

    try:
        outputs, commands, resolved = [], [], set()
        with tempfile.TemporaryDirectory() as temporary:
            for size in sizes:
                sheet = output / f"top{args.k}-{args.split}-n{size}.csv"
                summary = Path(temporary) / f"n{size}.json"
                cmd = [str(binary), "--database", str(database), "--queries", str(queries),
                       "--database-size", str(size), "--k", str(args.k),
                       "--route", "optimized_flat", "--param", f"backend={args.backend}",
                       "--warmup-queries", "0", "--repetitions", "1", "--threads", "1",
                       "--batch-repetitions", "1",
                       "--results-output", str(sheet), "--output", str(summary)]
                commands.append(shlex.join(cmd))
                print(f"n={size}", flush=True)
                result = subprocess.run(cmd, capture_output=True, text=True)
                if result.returncode != 0:
                    raise RuntimeError(f"answer sheet for n={size} failed:\n{result.stderr}")
                details = json.loads(summary.read_text())
                if details["candidates_per_query"] != size or details["query_rows"] != query_rows:
                    raise RuntimeError(f"n={size} did not scan every vector for every query")
                resolved.add(details["route_metadata"]["backend"])
                outputs.append({"file": sheet.name, "database_rows": size, "queries": query_rows,
                                "k": args.k, "sha256": sha256(sheet)})

        if len(resolved) != 1:
            raise RuntimeError(f"backend changed between sizes: {sorted(resolved)}")
        record = {
            "schema_version": 1,
            "generator": "scripts/make_references.py",
            "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
            "method": "exhaustive optimized_flat scan, ties broken by smaller database ID",
            "backend_requested": args.backend,
            "backend": resolved.pop(),
            "split": args.split,
            "k": args.k,
            "database_file": database.name,
            "database_sha256": database_sha,
            "queries_file": queries.name,
            "queries_sha256": queries_sha,
            "dataset_manifest_sha256": sha256(manifest_path),
            "outputs": outputs,
            "binary_sha256": sha256(binary),
            "git": git_state(),
            "freeze_record": freeze,
            "commands": commands,
        }
        (output / "reference-manifest.json").write_text(json.dumps(record, indent=2) + "\n")
        print(f"Saved {len(outputs)} answer sheets ({record['backend']}) in {output}")
        return 0
    except (OSError, ValueError, KeyError, RuntimeError) as error:
        shutil.rmtree(output, ignore_errors=True)
        parser.error(str(error))


if __name__ == "__main__":
    sys.exit(main())
