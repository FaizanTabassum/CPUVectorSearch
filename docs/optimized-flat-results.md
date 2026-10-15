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
