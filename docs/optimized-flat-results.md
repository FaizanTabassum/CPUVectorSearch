# Optimized flat layout and SIMD ablation

## Method

This is a B-specific development ablation, not the final project evaluation. A single
seeded synthetic 100k × 50 database was prepared with seed 5206; the 10k and 50k cases are
prefixes of that same database. All cases use the same 25 tuning queries and `k=10`.

For each size, three separate process runs were collected. Each process warmed up five
queries, measured five full query passes (125 timed searches), then measured throughput in
a separate untimed-per-query pass. Reported latency and throughput are medians across the
three process runs; p95 is the median of the three per-process nearest-rank p95 values.
Index build time is the median of the three process builds.

The machine was an Intel Core i5-10300H under Linux x86_64, GCC 13.3.0, Release build. The
three routes are scalar flat, blocked storage with scalar scoring, and blocked storage with
the automatically selected AVX2 backend. Search timing includes query validation, scoring,
top-k selection, allocation, and result ordering. It excludes input loading and index build.

## Results

| Database rows | Scalar median / p95 (ms) | Blocked scalar median (ms) | AVX2 median / p95 (ms) | AVX2 speedup vs scalar | AVX2 batch QPS | AVX2 build (ms) | Padded index storage | ID-changing queries | Max ranked-score delta |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 10,000 | 0.511 / 0.810 | 0.443 | 0.227 / 0.345 | 2.25× | 4,488 | 1.191 | 2.24 MB | 0 / 25 | 3.58e-7 |
| 50,000 | 3.312 / 4.553 | 2.965 | 1.597 / 2.541 | 2.07× | 621 | 6.667 | 11.2 MB | 0 / 25 | 3.58e-7 |
| 100,000 | 5.637 / 7.404 | 5.837 | 3.139 / 4.351 | 1.80× | 267 | 13.325 | 22.4 MB | 0 / 25 | 2.38e-7 |

The blocked layout alone was close to scalar flat: it helped at 10k and 50k in this run, but
was about 3.5% slower at 100k. AVX2 scoring improved median latency at all three sizes.
Across the 25-query synthetic set, AVX2 changed no neighbor IDs. Its maximum ranked-score
difference from scalar was below `3.6e-7`, consistent with the documented FP32 reduction
order difference. This does not establish behavior on real GloVe near-ties.

Padded index storage is the additional coordinate copy only; it is not peak process memory.
The source database remains separately resident. The synthetic data and raw JSON runs live
in ignored local output folders and are not part of this report's tracked artifacts.

## Real GloVe validation

The real-data runs use the Stanford GloVe 2024 Wikipedia + Gigaword, 50-dimensional
archive. The local preparation selected 100,000 database vectors, 500 tuning queries, and
1,000 held-out queries with seed 5206. The 10k and 50k databases are prefixes of that same
shuffled database. All routes used `k=10`. The three routes and timing protocol match the
synthetic ablation above; each workload was run in three separate processes. Table values
are medians across processes, and p95 is the median of each process's p95.

| Database rows | Scalar median / p95 (ms) | Blocked scalar median (ms) | AVX2 median / p95 (ms) | AVX2 speedup | AVX2 batch QPS | AVX2 build (ms) | Padded index storage | Peak process RSS | ID-changing queries | Max ranked-score delta |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 10,000 | 0.370 / 0.629 | 0.380 | 0.148 / 0.322 | 2.50× | 5,471 | 1.202 | 2.24 MB | 9.8 MiB | 0 / 500 | 3.58e-7 |
| 50,000 | 2.110 / 3.389 | 2.082 | 1.033 / 1.619 | 2.04× | 839 | 6.330 | 11.2 MB | 34.6 MiB | 0 / 500 | 3.58e-7 |
| 100,000 | 4.037 / 4.808 | 3.908 | 2.030 / 2.802 | 1.99× | 474 | 10.580 | 22.4 MB | 65.5 MiB | 0 / 500 | 3.58e-7 |

AVX2 returned the same top-ten IDs as scalar for all 500 tuning queries at all three sizes.
Its FP32 scores differed by at most `3.58e-7` from the scalar route. Peak process RSS was
measured in a separate one-pass run with `/usr/bin/time -v`; it includes the loaded database,
queries, executable, and index. The padded index copy alone uses 22.4 MB at 100k rows, and
the measured full-process peak was 65.5 MiB, below the project's 512 MB search-engine
budget. These are tuning results, not held-out evaluation results.

The source archive SHA-256 is
`afa5e258ee38272db6394547c4b075ecbb7b2164e98542c8d1237b6029b35a65`. The generated
reference manifest records the database and query hashes. Dataset files, generated
reference answers, and raw run JSON remain under ignored local `data/` and `results/`
folders. The nine timed-run records are named
`results/raw/b-flat-real-n{10000,50000,100000}-run-{1,2,3}.json`.

## Reproduce

Prepare the seeded 100k database and 25-query tuning split:

```sh
python3 scripts/prepare_data.py --synthetic \
  --database-size 100000 --tuning-size 25 --test-size 1 \
  --dimensions 50 --seed 5206 --output data/processed/b-ablation
```

Create nested database prefixes by retaining the CVEC header and the first `N × D` FP32
coordinates, updating the header row count to `N`. Then run each prefix three times:

```sh
./build/flat_ablation \
  --database data/processed/b-ablation/database-prefix-10000.cvec \
  --queries data/processed/b-ablation/tuning.cvec \
  --k 10 --warmup-queries 5 --repetitions 5 --data-seed 5206 \
  --machine-model "Intel Core i5-10300H" \
  --output results/raw/b-flat-ablation-10000-run-1.json
```

Use the same command for the 50k and 100k prefix files and for runs 2 and 3. The JSON
records compiler, OS, architecture, workload shape, seed, index build/storage, latency,
throughput, and top-k differences versus scalar flat.

To reproduce the real-data workload, download and prepare the GloVe vectors using
`docs/getting-started.md`, then create the three database prefixes from the prepared
100k database. This script preserves the CVEC header format and coordinates:

```sh
python3 - <<'PY'
from pathlib import Path
import struct

folder = Path("data/processed/glove-50d-seed5206")
source = folder / "database.cvec"
with source.open("rb") as database:
    header = database.read(16)
    rows, dimensions = struct.unpack("<II", header[8:])
    for count in (10_000, 50_000, 100_000):
        database.seek(16)
        remaining = count * dimensions * 4
        with (folder / f"database-prefix-{count}.cvec").open("wb") as prefix:
            prefix.write(header[:8] + struct.pack("<II", count, dimensions))
            while remaining:
                block = database.read(min(1024 * 1024, remaining))
                if not block:
                    raise EOFError("truncated database")
                prefix.write(block)
                remaining -= len(block)
PY
```

Generate the scalar practice answers with `scripts/generate_reference.py`, then run the
same command shown above with each prefix and three output names. The measured run used 500
tuning queries, five warmup queries, and five timed passes per process. For peak RSS, use a
separate process with `--warmup-queries 0 --repetitions 1` wrapped in `/usr/bin/time -v`.
