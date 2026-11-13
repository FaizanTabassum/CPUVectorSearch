# Final held-out results (frozen settings)

This folder holds the final measurement on the 1,000 held-out test queries, run once with the
settings in `configs/evaluation-freeze.json` (frozen on 11 November 2026). Nothing was tuned after
these results were produced.

## How it was produced

On the agreed machine (Intel Core i7-10750H, Windows, GCC 16.2 from MSYS2, Release build,
`-ffp-contract=off`, mains power, best-performance mode), following `docs/evaluation-freeze.md`:

```sh
./build/vector_search --database data/processed/glove-50d-seed5206/database.cvec --queries data/processed/glove-50d-seed5206/tuning.cvec --database-size 10000 --route optimized_flat --param backend=avx2 --k 10 --threads 1 --warmup-queries 0 --repetitions 1 --batch-repetitions 1 --output results/raw/final-machine-preflight.json
python3 scripts/make_references.py --split test --backend scalar --config configs/experiments-frozen.json --output data/processed/glove-50d-seed5206/reference-test-scalar
python3 scripts/make_references.py --split test --backend avx2 --config configs/experiments-frozen.json --output data/processed/glove-50d-seed5206/reference-test-avx2
python3 scripts/run_benchmarks.py --split test --strict --config configs/experiments-frozen.json --routes scalar_flat --reference-dir data/processed/glove-50d-seed5206/reference-test-scalar --output results/final/frozen-2026-11-11/scalar
python3 scripts/run_benchmarks.py --split test --strict --config configs/experiments-frozen.json --routes optimized_flat ivf --optimized-backends avx2 --ivf-backend avx2 --reference-dir data/processed/glove-50d-seed5206/reference-test-avx2 --output results/final/frozen-2026-11-11/avx2
```

The two `summary.csv` files were then joined into `combined/summary.csv` (45 rows) and
`python3 scripts/plot_results.py results/final/frozen-2026-11-11/combined` made the tables and
figures.

- Routes: `scalar_flat` (scalar), `optimized_flat` (AVX2), `ivf` with 256 lists, 64 probes,
  25 k-means iterations and seeds 5206, 5207 and 5208 (AVX2).
- Database sizes: 10,000, 50,000 and 100,000 vectors.
- Each route is scored against answer sheets made with its own backend.
- 10 warmup queries, 5 timed passes, 3 batch passes at 1, 2 and 4 threads, 3 separate processes.

## Files

| Path | Contents |
| --- | --- |
| `final-machine-preflight.json` | Preflight on the tuning queries: AVX2 backend, machine and build checks |
| `references-scalar/`, `references-avx2/` | Checksum manifests of the test answer sheets |
| `scalar/`, `avx2/` | `summary.csv`, `run-manifest.json` and every run's JSON and per-query CSV in `runs/` |
| `combined/summary.csv` | All 45 runs in one table |
| `combined/report/summary.md` | Latency, recall, throughput, memory and IVF tables |
| `combined/report/aggregated.csv` | Medians over processes, IVF averaged over seeds |
| `combined/report/*.png` | Recall against latency, latency by size, thread scaling, memory, build time |

## Main results

| Size | Route | Recall@10 | Median ms | p95 ms | 4-thread q/s |
| ---: | --- | ---: | ---: | ---: | ---: |
| 10,000 | scalar_flat | 1.000 | 0.565 | 0.628 | 5,329 |
| 10,000 | optimized_flat | 1.000 | 0.246 | 0.266 | 11,871 |
| 10,000 | ivf 256/64 | 0.965 | 0.146 | 0.168 | 20,206 |
| 50,000 | scalar_flat | 1.000 | 2.841 | 3.170 | 1,075 |
| 50,000 | optimized_flat | 1.000 | 1.310 | 1.492 | 2,155 |
| 50,000 | ivf 256/64 | 0.978 | 0.710 | 0.838 | 3,965 |
| 100,000 | scalar_flat | 1.000 | 5.907 | 6.523 | 538 |
| 100,000 | optimized_flat | 1.000 | 2.832 | 3.097 | 1,007 |
| 100,000 | ivf 256/64 | 0.981 | 1.940 | 2.282 | 1,368 |

IVF reached the 0.95 recall target on the held-out queries at every size (worst seed 0.962).
