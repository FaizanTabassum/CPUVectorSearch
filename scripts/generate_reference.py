#!/usr/bin/env python3
"""Generate checksummed scalar reference answers for the tuning split only."""

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import platform
import struct
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def checked_input(folder, manifest, name):
    split = manifest["splits"][name]
    filename = split["file"]
    if Path(filename).name != filename:
        raise ValueError("split files must be immediate children of the dataset directory")
    path = folder / filename
    if sha256(path) != split["sha256"]:
        raise ValueError(f"{name} checksum does not match dataset manifest")
    with path.open("rb") as source:
        header = source.read(16)
    if len(header) != 16 or header[:8] != b"CVEC0001":
        raise ValueError(f"invalid {name} CVEC header")
    rows, dimensions = struct.unpack("<II", header[8:])
    if rows != split["rows"] or dimensions != manifest["dimensions"] or dimensions == 0:
        raise ValueError(f"{name} shape does not match manifest")
    if path.stat().st_size != 16 + rows * dimensions * 4:
        raise ValueError(f"{name} payload size does not match header")
    ids = split["source_row_ids"]
    if rows == 0 or len(ids) != rows or len(set(ids)) != rows:
        raise ValueError(f"{name} needs nonempty, unique source row IDs")
    return path


def validate_answers(path, query_rows, database_rows, k):
    with path.open(newline="") as source:
        reader = csv.DictReader(source)
        if reader.fieldnames != ["query_id", "rank", "neighbor_id", "squared_distance"]:
            raise ValueError("unexpected reference CSV columns")
        position = 0
        previous = None
        seen = set()
        for row in reader:
            query, rank, neighbor = (int(row[name]) for name in ("query_id", "rank", "neighbor_id"))
            distance = float(row["squared_distance"])
            if (query, rank) != (position // k, position % k + 1) or query >= query_rows:
                raise ValueError("reference answers have wrong query/rank sequence")
            if rank == 1:
                previous, seen = None, set()
            key = (distance, neighbor)
            if not math.isfinite(distance) or distance < 0 or not 0 <= neighbor < database_rows:
                raise ValueError("invalid reference neighbor or distance")
            if neighbor in seen or (previous is not None and key < previous):
                raise ValueError("reference neighbors repeat or violate distance/ID ordering")
            previous = key
            seen.add(neighbor)
            position += 1
    if position != query_rows * k:
        raise ValueError("reference answer count does not match query count x k")
    return position


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--binary", type=Path, default=ROOT / "build/reference_search")
    parser.add_argument("--sizes", type=int, nargs="+", default=[10000, 50000, 100000])
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--output", type=Path, help="new directory; defaults to DATASET/reference-tuning")
    args = parser.parse_args(argv)
    try:
        folder, binary = args.dataset.resolve(), args.binary.resolve()
        output = (args.output or folder / "reference-tuning").resolve()
        if output.exists():
            raise ValueError("output already exists; choose a new directory")
        if not binary.is_file():
            raise ValueError("reference_search executable not found; build it first")
        manifest_path = folder / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        if manifest["format"] != "CVEC0001":
            raise ValueError("unsupported dataset format")
        database = checked_input(folder, manifest, "database")
        queries = checked_input(folder, manifest, "tuning")
        if set(manifest["splits"]["database"]["source_row_ids"]) & set(manifest["splits"]["tuning"]["source_row_ids"]):
            raise ValueError("database and tuning source rows overlap")
        if len(set(args.sizes)) != len(args.sizes) or not args.sizes:
            raise ValueError("subset sizes must be unique")
        if args.k <= 0 or any(n < args.k or n > manifest["splits"]["database"]["rows"] for n in args.sizes):
            raise ValueError("need 0 < k <= each subset size <= database rows")
        report = {
            "schema_version": 1, "query_split": "tuning", "route": "scalar_flat",
            "metric": "squared_l2", "arithmetic": "FP32 scalar; FP contraction disabled in CMake",
            "tie_break": "ascending_database_id", "id_base": 0, "rank_base": 1,
            "k": args.k, "dimensions": manifest["dimensions"],
            "query_rows": manifest["splits"]["tuning"]["rows"],
            "subset_policy": "prefix of the existing shuffled database; recompute each subset",
            "data_seed": manifest["seed"], "dataset_manifest_sha256": sha256(manifest_path),
            "database_sha256": manifest["splits"]["database"]["sha256"],
            "queries_sha256": manifest["splits"]["tuning"]["sha256"],
            "executable_sha256": sha256(binary), "source_sha256": {}, "outputs": []
        }
        report["host_platform"] = platform.platform()
        report["build_configuration"] = None
        cache = binary.parent / "CMakeCache.txt"
        if cache.is_file():
            settings = {}
            wanted = {"CMAKE_CXX_COMPILER", "CMAKE_BUILD_TYPE", "CMAKE_CXX_FLAGS",
                      "CMAKE_CXX_FLAGS_RELEASE", "CMAKE_CXX_FLAGS_DEBUG", "CVS_SANITIZERS"}
            for line in cache.read_text().splitlines():
                if "=" in line and ":" in line.split("=", 1)[0]:
                    key, value = line.split("=", 1)
                    key = key.split(":", 1)[0]
                    if key in wanted:
                        settings[key] = value
            report["build_configuration"] = settings
            if settings.get("CMAKE_CXX_COMPILER"):
                compiler = subprocess.run([settings["CMAKE_CXX_COMPILER"], "--version"],
                                          capture_output=True, text=True, check=True)
                report["compiler_version"] = compiler.stdout.strip()
        for name in ["apps/ground_truth.cpp", "scripts/generate_reference.py", "src/search.cpp",
                     "src/io.cpp", "include/cvs/search.hpp", "include/cvs/io.hpp", "CMakeLists.txt"]:
            report["source_sha256"][name] = sha256(ROOT / name)
        revision = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True)
        report["base_git_commit"] = revision.stdout.strip() if revision.returncode == 0 else None
        # Source hashes above identify local edits even though this work is uncommitted.
        report["source_hashes_identify_local_changes"] = True
        output.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix=".reference-", dir=output.parent) as temporary:
            staging = Path(temporary) / "answers"
            staging.mkdir()
            for size in args.sizes:
                filename = f"top{args.k}-tuning-n{size}.csv"
                path = staging / filename
                command = [str(binary), "--database", str(database), "--queries", str(queries),
                           "--database-size", str(size), "--k", str(args.k), "--output", str(path)]
                subprocess.run(command, check=True)
                count = validate_answers(path, report["query_rows"], size, args.k)
                report["outputs"].append({"file": filename, "database_rows": size,
                                           "answer_rows": count, "sha256": sha256(path)})
            (staging / "reference-manifest.json").write_text(json.dumps(report, indent=2) + "\n")
            if output.exists():
                raise ValueError("output appeared during generation; refusing to replace it")
            staging.rename(output)
        print(f"Validated tuning reference answers saved in {output}")
        return 0
    except (OSError, ValueError, KeyError, TypeError, subprocess.CalledProcessError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
