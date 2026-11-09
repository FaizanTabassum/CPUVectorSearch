# IVF tuning results (500 practice queries)

This folder holds the tuning run used to choose the IVF settings for the evaluation freeze.
Only the 500 tuning queries were used. The 1,000 held-out test queries were not touched.

## How it was produced

```sh
python3 scripts/restore_prepared_data.py
python3 scripts/run_benchmarks.py --output results/raw/tuning --routes scalar_flat optimized_flat ivf --threads 1 --process-runs 1 --repetitions 3 --batch-repetitions 1
python3 scripts/plot_results.py results/raw/tuning
```

- Database sizes: 10,000, 50,000 and 100,000 vectors.
- Routes: `scalar_flat`, `optimized_flat`, and `ivf` with 64 and 256 lists, every probe count from
  `configs/experiments.json`, and training seeds 5206, 5207 and 5208.
- Answer sheets: A's scalar tuning answer sheets (`reference-tuning/`). The AVX2 answer sheets made
  with `scripts/make_references.py` gave the same top-10 for every tuning query at all three sizes,
  so the choice of scoring backend does not change these recall numbers.
- Machine: Intel Core i7-10750H, Windows, GCC 16.2 (MSYS2), Release build, on mains power.
- Single-thread latency; 3 timed passes; one process per configuration.

This is a tuning run, not the final measurement. The final run on the held-out queries uses the
full repetition policy in `docs/benchmarking.md`.

## Files

| File | Contents |
| --- | --- |
| `summary.md` | Tables: latency, recall, throughput, memory, and the full IVF sweep |
| `summary.csv` | One row per configuration, straight from `run_benchmarks.py` |
| `aggregated.csv` | Medians per configuration, IVF averaged over seeds |
| `run-manifest.json` | Git commit, compiler, CPU, dataset and answer-sheet hashes, every command |
| `*.png` | Recall against latency, latency by size, thread scaling, memory, build time |

## How to read it

For each database size, `summary.md` shows the fastest IVF setting whose seed-averaged
recall@10 is at least 0.95. C uses this, together with the full sweep table, to propose the
settings that go into `configs/evaluation-freeze.json`.
