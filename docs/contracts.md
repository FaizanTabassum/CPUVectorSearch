# Shared implementation contracts

These are initial decisions for the team's week-1 review. Change contracts together and
update tests before integrating a different layout or route.

## Representation and identifiers

- `DenseMatrix` owns contiguous row-major FP32 coordinates. The project dataset has
  50 dimensions; the API accepts any positive dimension for small test fixtures.
- Prepared vectors have unit L2 norm (allow FP32 rounding); zero and non-finite vectors
  are rejected by preparation. The matrix/loader enforce shape and finiteness but do
  not enforce unit norm, so small correctness fixtures may use arbitrary finite values.
- A database vector's stable ID is its zero-based row in `database.cvec`. Its original
  source row is in the manifest; database and query IDs are different namespaces.
- Keep this database ID when building blocked layouts and inverted lists. IDs in a
  smaller prefix subset retain their values. Never use an inverted-list position as an ID.
- The three data splits contain disjoint original source row IDs. Centroid training uses
  database data only; tuning and test queries never participate in training.

## CVEC0001 file format

All multi-byte numbers use little-endian byte order. There is no padding.

| Offset | Type | Meaning |
| --- | --- | --- |
| 0 | 8 ASCII bytes | `CVEC0001` magic and format version |
| 8 | uint32 | Number of rows N (zero allowed by loader) |
| 12 | uint32 | Positive dimensions D |
| 16 | N * D IEEE-754 binary32 | Row-major coordinates |

The exact size is `16 + N * D * 4` bytes. The reader rejects wrong magic/version,
truncation, trailing bytes, invalid dimensions, and non-finite coordinates. IDs are
implicit row positions, so tokens are not stored in the binary payload.

## Search interface

`SearchIndex::search(query, k)` returns `SearchResult` with neighbours and a candidate count.
The database must outlive an index and remain immutable. A call owns all query scratch
space; future threading must not add unsynchronised mutable shared scratch state.

- Compute squared L2 with FP32 coordinates and FP32 scalar accumulation initially.
  Do not take a square root. Unit vectors make squared L2 compatible with cosine ranking.
- Return ascending `(squared_distance, database_id)` order; smaller IDs win exact FP32 ties.
- Return `min(k, available_candidates)` neighbours with distinct database IDs.
- `k == 0` returns an empty result with zero candidates. Empty databases return no results.
- A query must match the database dimensions and contain finite values, even when `k == 0`.
- Count vectors whose distance to the query was evaluated, not centroid comparisons.
  Exhaustive search evaluates N candidates for positive k and nonempty databases.
- The benchmark requires positive k <= N and nonempty queries/database. Library edge
  cases are broader than benchmark inputs.

The scalar route uses O(N * D + N log k) query work and O(k) selection storage. A future
optimised scan and IVF route must use a shared distance/selection kernel for the all-lists
equivalence check. No fake optimised/IVF aliases exist in this starter.

## Numerical correctness

`tests/search_tests.cpp` implements a separate FP64 distance calculation and full sort.
It checks IDs on exact dyadic fixtures and a small normalised fixture, FP32 score tolerance,
ties, k boundaries, invalid input, and empty databases. The reference does not call the
production distance function, ordering helper, or heap.

FP32 and FP64 can rank extremely close distances differently. Investigate and document
near-ties instead of claiming universal bitwise agreement with FP64. The baseline disables
FP contraction; do not enable fast-math globally. Document SIMD reduction changes and
their effects on rankings. Final recall ground truth comes from the shared exhaustive
kernel, separately checked against the small FP64 oracle.

Before IVF is accepted, require:

1. Probing every list returns the same IDs and scores as exhaustive search using the
   same kernel, including ties and duplicate vectors.
2. Candidate counts agree with the union of probed lists; no candidate is counted twice.
3. Invalid probe counts and dimensions are rejected, and empty lists are safe.
4. Training and assignments are reproducible for each recorded training seed.

## Measurement and freeze rules

The current CLI is a smoke harness, not the final experiment runner. It runs one thread,
warms up on up to ten queries, and measures one pass over the chosen query split.
Latency includes query validation, distance evaluation, top-k selection, result allocation
and sorting. File loading, index creation, query-buffer preparation, warmup, result
aggregation, and output writing are outside query timing. Report load/build times separately.
Percentiles use the nearest-rank convention; the median is the 50th-percentile rank.
Batch throughput uses a separate sequential pass without per-query timer calls and includes
a small result-consumption loop to keep outputs observable.

The final harness must add:

- Recall@10 as the mean per-query intersection size with exhaustive top ten divided by ten.
  Recompute exhaustive ground truth for each database subset. Do not fill in recall as
  1.0 without actually evaluating results.
- All IVF query overhead, including centroid scoring/probe selection and candidate gathering.
- Separate construction/training time, candidate counts, and peak memory with a documented
  measurement method. Database-vector bytes alone are not peak memory.
- Raw per-query measurements and run metadata: commit, compiler/options, CPU/OS, dimensions,
  sizes, split hashes, seeds, route, list/probe counts, thread counts, and warmup/repetition policy.
- Representative batch throughput at 1/2/4 threads capped by available hardware. Keep core
  latency comparisons at one thread and control machine load.
- Frozen tuning decisions before held-out evaluation. Member A records the freeze; results
  must preserve that configuration and identify any later exploratory run explicitly.

The planned matrix is in `configs/experiments.json`; no program consumes it yet. The final
team should add repeat policy and machine metadata before treating runs as publishable evidence.
