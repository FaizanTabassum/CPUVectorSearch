# Benchmarking and measurement procedure (Member D)

This page explains how we measure the three search routes, what every number means, and
the exact procedure for the tuning runs and the final held-out run. It is D's input to the
evaluation freeze in `configs/evaluation-freeze.json` (see [evaluation-freeze.md](evaluation-freeze.md)).

## The tools

| Tool | What it does |
| --- | --- |
| `build/vector_search` | Measures one route on one database size and writes a JSON summary |
| `scripts/run_benchmarks.py` | Runs the whole grid from `configs/experiments.json`, one process per configuration |
| `scripts/plot_results.py` | Turns a run folder into `summary.md`, `aggregated.csv` and five figures |

Figures need matplotlib (`python3 -m pip install matplotlib`). Without it, the tables are
still written.

## What is measured

**Recall@10.** For each query, the number of returned IDs that are also in A's exact top ten
for the same database size, divided by ten. We report the mean over queries, the worst
query and the number of perfect queries. It is always computed from returned IDs. Without
an answer sheet the value is `null`.

**Latency.** One `steady_clock` reading before and after each `search()` call, on one
thread. It includes query validation, scoring, top-k selection and building the result.
For IVF it also includes centroid scoring and gathering the probed lists. Loading files,
building the index and warmup are not included. We report mean, median, p95, p99 and max,
using the nearest-rank rule.

**Batch throughput.** Queries per second for a whole pass over the query set, with no
per-query clock reads, at 1, 2 and 4 threads. Thread counts above the machine's hardware
threads are skipped. Threads take queries from a shared counter. After timing, the queries
are run again with several threads and compared with the single-thread results
(`threaded_results_match`). Core latency comparisons always use one thread.

**Candidates.** Database vectors actually scored per query (`candidates_per_query`).
Centroid comparisons are not counted.

**Build time.** Time to construct the index. For IVF this includes k-means training.

**Memory.** Resident memory of the benchmark process:

| OS | Method | Notes |
| --- | --- | --- |
| Linux | `VmRSS` and `VmHWM` from `/proc/self/status` | Peak reset after loading through `/proc/self/clear_refs`, so it excludes the file-reading spike |
| Windows | `GetProcessMemoryInfo` (`WorkingSetSize`, `PeakWorkingSetSize`) | Peak cannot be reset, so it includes loading |
| macOS | `getrusage` `ru_maxrss` | Whole-process peak only |

`index_rss_delta_bytes` is the memory added by building the index. The JSON records which
method was used, so results from different machines are not mixed up by mistake.

## Repetition policy

These values are in `configs/experiments.json` and are read by `run_benchmarks.py`:

| Setting | Value |
| --- | --- |
| Warmup | first 10 queries searched once, untimed |
| Timed passes per process | 5 |
| Batch passes per thread count | 3, median reported |
| Separate processes per configuration | 3, interleaved |
| Thread counts | 1, 2, 4 (capped by hardware) |
| k | 10 |

Tables use the median across the three processes. IVF numbers are then averaged over the
three training seeds, and the seed range is shown next to the mean.

## Machine rules

- Release build (`cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release`).
- Laptop on mains power, high-performance power mode, other applications closed.
- All runs for one comparison on the same machine. The CPU model, OS and compiler are
  recorded in every JSON file and in `run-manifest.json`.
- Final runs use: Intel Core i7-10750H, Windows, GCC 16.2 (MSYS2 MinGW-w64), unless the team
  agrees on another machine.

## Answer sheets and scoring backend

A's answer sheets use scalar scoring. The AVX2 backend adds numbers in a different order,
so near-ties can rank differently. Two rules:

- For exact checks (IVF with every list probed must match exhaustive search), use the same
  backend on both sides: `--ivf-backend scalar` against A's scalar sheets.
- For recall of AVX2 routes, an AVX2 exhaustive answer sheet can be made with the runner
  itself: `--route optimized_flat --param backend=avx2 --results-output top10-avx2.csv`.
  Record which sheet was used.

## Tuning procedure (500 practice queries)

```sh
python3 scripts/restore_prepared_data.py
python3 scripts/run_benchmarks.py --split tuning --output results/raw/tuning-<date>
python3 scripts/plot_results.py results/raw/tuning-<date>
```

Without options this runs the full planned grid: 10k, 50k and 100k vectors; all three
routes; IVF with 64 and 256 lists, every probe count up to the list count, and seeds
5206, 5207 and 5208. For a quicker IVF-only sweep:

```sh
python3 scripts/run_benchmarks.py --output results/raw/ivf-tuning --routes ivf --threads 1 --process-runs 1
```

`summary.md` lists, for each size, the fastest IVF setting whose seed-averaged recall@10
is at least 0.95. C uses this to propose the settings for the freeze.

## Final procedure (1,000 held-out queries)

1. A fills in and commits `configs/evaluation-freeze.json` with `"status": "frozen"`.
2. A generates the held-out answer sheets with the agreed backend in
   `data/processed/glove-50d-seed5206/reference-test/`.
3. D runs only the frozen settings:

```sh
python3 scripts/run_benchmarks.py --split test \
  --reference-dir data/processed/glove-50d-seed5206/reference-test \
  --routes scalar_flat optimized_flat ivf \
  --ivf-lists <frozen> --ivf-probes <frozen> \
  --output results/final/test-<date>
python3 scripts/plot_results.py results/final/test-<date>
```

`run_benchmarks.py` refuses `--split test` while the freeze record is not frozen.

## Where results are stored

Each run folder contains:

| File | Contents |
| --- | --- |
| `runs/<name>.json` | Summary for one configuration |
| `runs/<name>-queries.csv` | Every timed search: repetition, query, latency, candidates, recall |
| `summary.csv` | One row per configuration and process run |
| `run-manifest.json` | Git commit, dirty flag, binary hash, build settings, CPU, dataset and answer-sheet hashes, config hash, freeze-record hash, every command |
| `report/` | Tables and figures from `plot_results.py` |

`results/raw/` is ignored by Git. Tuning and final folders that go into the report are
copied into a versioned submission folder, not left in `results/raw/`.

## For the freeze record

Suggested `timing_protocol` entry for `configs/evaluation-freeze.json`:

```json
{
  "tool": "scripts/run_benchmarks.py",
  "warmup_queries": 10,
  "timed_passes": 5,
  "batch_passes": 3,
  "process_runs": 3,
  "threads": [1, 2, 4],
  "core_latency_threads": 1,
  "memory_method": "process resident memory; Windows PeakWorkingSetSize",
  "machine": "Intel Core i7-10750H, Windows, GCC 16.2, Release build, mains power",
  "raw_results": "versioned submission folder copied from results/final/"
}
```
