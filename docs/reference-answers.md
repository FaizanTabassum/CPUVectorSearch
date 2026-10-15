# Member A: reference answer sheets

An answer sheet records the ten database entries numerically closest to each input query.
The generator uses `ScalarFlatIndex` and checks every vector in the selected database
prefix. It never trains a model, chooses IVF settings, or times other routes.

## Build and generate practice answers

From the repository root, with the prepared GloVe directory already present:

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel 2
ctest --test-dir build --output-on-failure

python3 scripts/generate_reference.py \
  --dataset data/processed/glove-50d-seed5206 \
  --binary build/reference_search
```

The wrapper generates answers for all 500 **tuning** queries separately against the first
10,000, 50,000, and 100,000 database vectors. It does not read `test.cvec`. Held-out
generation/evaluation belongs after the settings freeze and is outside this tuning command.
The output directory must not already exist; use `--output` with a new path for another
run. Existing answers and dataset files are preserved.

Outputs under `data/processed/glove-50d-seed5206/reference-tuning/`:

- `top10-tuning-n10000.csv`
- `top10-tuning-n50000.csv`
- `top10-tuning-n100000.csv`
- `reference-manifest.json`: input/output hashes, subset sizes, source fingerprints and
  executable hash. The Git commit is the base commit; source hashes identify local edits.
  When a CMake cache is beside the binary, compiler version and build settings are recorded.

The wrapper checks input hashes and disjoint database/tuning source rows, validates output
shape, ID ranges, unique neighbours and ordering, and publishes only after every subset
succeeds. Vectors and generated answers remain in ignored local data directories.

## Reading an answer sheet

| Column | Meaning |
| --- | --- |
| `query_id` | Zero-based row in `tuning.cvec`; not a database ID |
| `rank` | Position 1 through 10, closest first |
| `neighbor_id` | Zero-based row in `database.cvec`, also valid in that prefix subset |
| `squared_distance` | FP32 squared-L2 distance; smaller is closer |

Equal FP32 distances are ordered by smaller database ID. Scores retain enough digits to
recover their FP32 values. These are nearest entries by vector distance; that does not
establish human semantic relevance. Keep the dataset size and manifest with each CSV.

Member D can group by `query_id` and compare a route's ten returned IDs with the ten
reference IDs. Nine shared IDs means recall@10 = 0.9; average over all queries. Distance
agreement alone is not neighbour recall.

## Correctness and later integration

`reference_answers` tests cover a hand-calculated example with duplicate vectors and
ties, an independent FP64 full-sort oracle on small normalised 50-dimensional inputs,
prefix sizes with different answers, reproducible output hashes, corruption rejection,
and tuning generation without the held-out vector file. Existing scalar oracle tests
remain enabled. FP32/FP64 can rank near-ties differently; investigate such cases rather
than claiming universal bitwise agreement.

These answers use the **scalar** backend. If the shared AVX2 backend changes floating-point
reduction order, identify it in metadata and regenerate its exhaustive answers before the
final recall comparison. IVF and exhaustive search must use the same scoring backend in
all-lists equivalence checks; the IVF route now accepts B's shared scorer for that purpose.
