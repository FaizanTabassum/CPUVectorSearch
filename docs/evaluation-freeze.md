# Frozen evaluation settings

The shared record is **`configs/evaluation-freeze.json`**. Member A completed the
pre-freeze checks on 11 November 2026. The freeze must be committed before any search
uses the 1,000 held-out inputs. No held-out searches were run during these checks.

## Decisions

| Setting | Frozen value |
| --- | --- |
| Database sizes | 10,000 / 50,000 / 100,000 |
| Neighbours returned | 10 |
| IVF groups / groups searched | 256 / 64 at every size |
| Training rounds | 25 |
| Training seeds | 5206, 5207, 5208; retain all three, do not pick the best seed afterwards |
| Scoring | Scalar baseline; AVX2 for optimized flat and IVF |
| Final machine | Aravind's Intel Core i7-10750H, Windows, GCC 16.2, Release, mains power |
| Warmup / timed passes | 10 queries / 5 passes |
| Batch / independent process repeats | 3 batch passes / 3 processes |
| Batch threads / latency threads | 1, 2, 4 / one thread |

`timing_protocol` is copied exactly from Aravind's **For the freeze record** block in
`docs/benchmarking.md`. `configs/experiments-frozen.json` narrows the planned search grid
to the chosen settings; keep `configs/experiments.json` as the original tuning grid.

Devam recommended 256/64 for recall margin: 64/16 fails at 10k and only narrowly passes
at 50k. The selected setting recovers 96.43%, 97.93%, and 98.27% of reference matches on
the practice inputs (mean over the three seeds). These are tuning results, not promises
about the final inputs. The selection uses only the 500 practice inputs.

## Evidence

- [Tuning results](../results/tuning/README.md), with all 150 summary rows and the run manifest.
- [Member A validation](../results/validation/pre-freeze-2026-11-11/README.md): pinned data hashes,
  disjoint source rows, all three reference prefixes, 13 passing test suites, nine exact
  all-lists IVF comparisons, and nine independent selected-setting recall checks.
- Devam's [recommendation and independent sweep](https://github.com/FaizanTabassum/CPUVectorSearch/issues/9#issuecomment-6784918896).

The original x86 tuning manifest records a dirty working tree and the raw per-process
files are not archived. This limitation is preserved in the freeze. Local ARM checks
independently reproduced all nine selected recall values; local timings are not used as
Intel/Windows performance evidence. Final measurements use the fuller repetition policy.

## Final-machine preflight: use practice inputs only

After the freeze commit is available on the measurement machine, build Release and run:

```sh
./build/vector_search --database data/processed/glove-50d-seed5206/database.cvec \
  --queries data/processed/glove-50d-seed5206/tuning.cvec --database-size 10000 \
  --route optimized_flat --param backend=avx2 --k 10 --threads 1 \
  --warmup-queries 0 --repetitions 1 --batch-repetitions 1 \
  --output results/raw/final-machine-preflight.json
```

Check that `route_metadata.backend` is **avx2**, the environment is Windows on the agreed
Intel CPU, and the build is Release with GCC 16.2 and FP contraction off. The current
scorer can fall back to scalar even when AVX2 was requested. **Stop if the resolved
backend or machine is wrong; do not run final inputs on the ARM validation machine.**
Record this preflight JSON with the final results.

## Commands after the freeze commit

Run these on the agreed machine only. Each output directory must be new. Make separate
reference answers for scalar and AVX2 arithmetic; do not assume their rankings remain
identical on previously unseen inputs.

```sh
python3 scripts/make_references.py --split test --backend scalar \
  --config configs/experiments-frozen.json \
  --output data/processed/glove-50d-seed5206/reference-test-scalar
python3 scripts/make_references.py --split test --backend avx2 \
  --config configs/experiments-frozen.json \
  --output data/processed/glove-50d-seed5206/reference-test-avx2

python3 scripts/run_benchmarks.py --split test --strict \
  --config configs/experiments-frozen.json --routes scalar_flat \
  --reference-dir data/processed/glove-50d-seed5206/reference-test-scalar \
  --output results/final/frozen-2026-11-11/scalar
python3 scripts/run_benchmarks.py --split test --strict \
  --config configs/experiments-frozen.json --routes optimized_flat ivf \
  --optimized-backends avx2 --ivf-backend avx2 \
  --reference-dir data/processed/glove-50d-seed5206/reference-test-avx2 \
  --output results/final/frozen-2026-11-11/avx2
```

The driver checks `status: frozen`, but does not automatically apply the record's
configuration or enforce the machine. Use the explicit frozen config and preflight above.
The pinned IVF route default is 25 training iterations; verify final route metadata still
reports 25. Preserve manifests, all raw per-process/per-query files, preflight and generated
reference manifests in the final submission archive. Do not tune after seeing final results.
If a change is needed, retain the original freeze and results and document the deviation.
