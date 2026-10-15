# CPUVectorSearch

CEG5206 project: **When does grouped CPU vector search beat an optimised exhaustive scan?**

We will compare a scalar exhaustive scan, an optimised exhaustive scan, and a group-built
inverted-file (IVF) index. The research target is at least **95% recall@10** with lower query
latency than the optimised exhaustive scan. A speedup is a hypothesis to test.

## What works now

- Dependency-free C++17 scalar and optimized flat squared-L2 search with bounded-heap top-k selection.
- Optimized flat storage with eight-row tiles, padded dimensions, AVX2 scoring where available, and a scalar fallback.
- A B-specific ablation runner comparing scalar flat, blocked scalar, and the selected optimized backend with repeated timings and ranking-difference counts.
- Shared index interface, row-major FP32 input storage, and deterministic distance/ID ordering.
- Independent FP64 oracle tests plus malformed-file and end-to-end checks.
- Seeded synthetic or local GloVe data preparation with disjoint source-row splits and checksums.
- Exhaustive reference answer sheets for tuning queries against nested database subsets,
  with input/output checksums and local source/build provenance.
- Single-thread smoke benchmark reporting load/build time, median/p95 latency, batch
  throughput, candidate count, and database-vector bytes as JSON.
- CMake/CTest and GitHub Actions for Linux/macOS Debug/Release and Linux sanitizers.

**Still to implement:** centroid training, IVF lists/probing, query threading, recall evaluation,
peak-memory measurement, the full experiment runner, and plots. The smoke benchmark reports
unimplemented metrics as `null` and currently runs only the scalar route.
Its database-vector byte count is not peak process or engine memory.

## Quick start

Requirements: a C++17 compiler, CMake 3.16+, and Python 3.9+ for data preparation/tests.
Python scripts use only the standard library. Install a compiler and CMake if absent
(for example, Apple Command Line Tools plus CMake on macOS, or GCC plus CMake on Linux).

From the repository root:

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel 2
ctest --test-dir build --output-on-failure

python3 scripts/prepare_data.py --synthetic \
  --database-size 1000 --tuning-size 20 --test-size 20 \
  --output data/processed/smoke

./build/vector_search \
  --database data/processed/smoke/database.cvec \
  --queries data/processed/smoke/tuning.cvec \
  --k 10 --output results/raw/smoke.json
```

Use `--help` on either benchmark program and `scripts/prepare_data.py` for options. Preparation refuses to overwrite an existing
dataset; choose a new output directory for another run. Synthetic timings only verify
that the pipeline works and are not evidence for the project's research question.

For B's layout/SIMD ablation, run the dedicated tool on a prepared CVEC dataset:

```sh
./build/flat_ablation \
  --database data/processed/smoke/database.cvec \
  --queries data/processed/smoke/tuning.cvec \
  --k 10 --warmup-queries 5 --repetitions 5 --data-seed 5206 \
  --output results/raw/flat-ablation-smoke.json
```

The JSON compares the scalar route, blocked storage with scalar scoring, and blocked storage
with the selected backend. It records build time, padded index bytes, repeated query latency,
throughput, and top-k ranking differences versus scalar. A supplied `--machine-model` label
is included in the metadata. This focused ablation does not replace the final experiment runner.

For a sanitizer build with Clang/GCC:

```sh
cmake -S . -B build-sanitize -DCMAKE_BUILD_TYPE=Debug -DCVS_SANITIZERS=ON
cmake --build build-sanitize --parallel 2
ctest --test-dir build-sanitize --output-on-failure
```

## Prepare the project dataset

Obtain and unpack the proposal's **Stanford GloVe 2024 Wikipedia/Gigaword, 50-dimensional**
text data into `data/raw/`. Confirm the release and its usage terms; do not substitute
another GloVe release without recording the change. The script accepts one UTF-8 token
plus 50 coordinates per line and rejects malformed, zero-norm, or non-finite rows.
It does not download data or train centroids.

```sh
python3 scripts/prepare_data.py \
  --glove data/raw/YOUR_50D_GLOVE_FILE.txt \
  --database-size 100000 --tuning-size 500 --test-size 1000 \
  --dimensions 50 --seed 5206 --output data/processed/glove-50d-seed5206
```

The script reservoir-samples 101,500 source rows, shuffles them with the recorded seed,
then makes separate database, tuning, and test splits. It L2-normalises before rounding
to FP32. `manifest.json` records source row IDs, input/output SHA-256 hashes, sizes,
seed, and Python version. Disjoint source rows may still contain identical vectors;
do not interpret row-disjointness as semantic deduplication.

Train centroids on database vectors only. Select probe counts using tuning queries,
then freeze all settings before running the held-out test queries. Use prefixes of
the shuffled database for nested 10k/50k/100k subsets and recompute exact neighbours
for each subset. The [reference answer workflow](docs/reference-answers.md) now generates
scalar top-ten answers for tuning queries against all three subsets. Full recall evaluation
and the held-out evaluation runner remain future tasks.

## Repository map

```text
apps/benchmark.cpp       Single-thread scalar smoke benchmark
apps/flat_ablation.cpp   B's layout/SIMD ablation benchmark
apps/ground_truth.cpp    Exhaustive reference answers saved as CSV
include/cvs/             Shared matrix, search, optimized-flat, and file-loading API
src/                     Scalar/optimized-flat search and CVEC reader
tests/                   Independent oracle, data and integration tests
scripts/prepare_data.py  Seeded sampling, normalisation, and split manifest
scripts/generate_reference.py  Tuning-only subset references with provenance
configs/experiments.json Planned final-study settings (not an executable runner)
docs/contracts.md        Data, search, correctness, and measurement contracts
docs/team-plan.md        Owners, milestones, review pairs, acceptance criteria
data/raw/                Local source data (ignored)
data/processed/          Local prepared datasets (ignored)
results/raw/             Local benchmark outputs (ignored)
results/plots/           Local plots (ignored)
```

## Start as a team

Read [the contracts](docs/contracts.md), assign members A-D using [the team plan](docs/team-plan.md),
and start all four workstreams in parallel from the existing interfaces. A prepares real
data and validation; B optimises flat search on synthetic fixtures; C builds IVF using the
existing scalar distance helpers; D extends benchmarking/threading against the working
scalar index. Nobody needs to wait for A's dataset or B's SIMD implementation to begin.

Run the quick start on each teammate's machine. Integrate real data and the shared optimised
kernel at the plan's checkpoints, preserving scalar correctness. Final comparisons and
held-out evaluation still require the completed routes, real inputs, and tuning freeze.

The final study targets an existing multicore CPU machine with 8 GB RAM, under 512 MB
search-engine memory, and roughly 3 GB disk for source data and outputs. It requires no
GPU, paid service, cloud database, or API. Quantisation is optional and outside this starter.

## Project basis

Scope follows the team's one-page proposal dated 2 October 2026. The proposal is not
redistributed here. Final deliverables are source, correctness tests, raw measurements,
plots, report, and a demonstration. Neighbour recall does not establish human semantic relevance.
