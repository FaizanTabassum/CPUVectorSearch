#!/usr/bin/env python3
"""Prepare disjoint, seeded database/tuning/test CVEC files using the stdlib."""

import argparse
from array import array
import hashlib
import json
import math
from pathlib import Path
import random
import struct
import sys

MAGIC = b"CVEC0001"


def normalize(values):
    norm = math.sqrt(sum(x * x for x in values))
    if not math.isfinite(norm) or norm == 0:
        raise ValueError("vectors must have a finite, nonzero norm")
    result = array("f", (x / norm for x in values))
    if not all(math.isfinite(x) for x in result):
        raise ValueError("normalization produced non-finite FP32 values")
    return result


def glove_sample(path, dimensions, count, rng):
    """Reservoir sample source rows, recording line IDs and the source checksum."""
    reservoir = []
    digest = hashlib.sha256()
    seen = 0
    with path.open("rb") as source:
        for line_number, raw in enumerate(source, 1):
            digest.update(raw)
            fields = raw.decode("utf-8").split()
            if len(fields) != dimensions + 1:
                raise ValueError(f"line {line_number}: expected token plus {dimensions} coordinates")
            values = [float(value) for value in fields[1:]]
            vector = normalize(values)
            seen += 1
            entry = (line_number, vector)
            if len(reservoir) < count:
                reservoir.append(entry)
            else:
                slot = rng.randrange(seen)
                if slot < count:
                    reservoir[slot] = entry
    if seen < count:
        raise ValueError(f"need {count} source rows; found {seen}")
    return reservoir, {"kind": "glove", "filename": path.name,
                       "sha256": digest.hexdigest(), "source_rows": seen}


def write_vectors(path, rows, dimensions):
    with path.open("wb") as output:
        output.write(MAGIC)
        output.write(struct.pack("<II", len(rows), dimensions))
        for _, vector in rows:
            payload = array("f", vector)
            if sys.byteorder != "little":
                payload.byteswap()
            output.write(payload.tobytes())


def file_sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--synthetic", action="store_true", help="smoke data, not scientific evidence")
    source.add_argument("--glove", type=Path, help="unpacked token + coordinate GloVe text file")
    parser.add_argument("--database-size", type=int, default=100000)
    parser.add_argument("--tuning-size", type=int, default=500)
    parser.add_argument("--test-size", type=int, default=1000)
    parser.add_argument("--dimensions", type=int, default=50)
    parser.add_argument("--seed", type=int, default=5206)
    parser.add_argument("--output", type=Path, default=Path("data/processed"))
    args = parser.parse_args(argv)
    counts = (args.database_size, args.tuning_size, args.test_size)
    if args.dimensions <= 0 or args.dimensions > 0xFFFFFFFF or any(n <= 0 or n > 0xFFFFFFFF for n in counts):
        parser.error("split sizes and dimensions must be positive uint32 values")
    # Avoid silently replacing a previous split or frozen experiment input.
    if args.output.exists() and any(p.name != ".gitkeep" for p in args.output.iterdir()):
        parser.error("output directory must be empty; select a new directory")
    rng = random.Random(args.seed)
    count = sum(counts)
    try:
        if args.synthetic:
            rows = [(i + 1, normalize([rng.gauss(0, 1) for _ in range(args.dimensions)]))
                    for i in range(count)]
            provenance = {"kind": "synthetic", "generator": "python-random-gauss"}
        else:
            rows, provenance = glove_sample(args.glove, args.dimensions, count, rng)
        rng.shuffle(rows)
        database_end = args.database_size
        tuning_end = database_end + args.tuning_size
        splits = {"database": rows[:database_end], "tuning": rows[database_end:tuning_end],
                  "test": rows[tuning_end:]}
        args.output.mkdir(parents=True, exist_ok=True)
        metadata = {}
        for name, split in splits.items():
            path = args.output / f"{name}.cvec"
            write_vectors(path, split, args.dimensions)
            metadata[name] = {"file": path.name, "rows": len(split),
                              "sha256": file_sha256(path), "source_row_ids": [i for i, _ in split]}
        manifest = {"format": "CVEC0001", "seed": args.seed, "dimensions": args.dimensions,
                    "normalization": "L2 unit norm, rounded to FP32", "source": provenance,
                    "python_version": sys.version.split()[0], "splits": metadata}
        (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        print(f"Prepared {counts} rows in {args.output}; seed={args.seed}")
    except (OSError, ValueError, OverflowError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
