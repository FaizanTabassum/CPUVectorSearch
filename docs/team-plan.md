# Team plan and acceptance milestones

**All four members start in parallel from the existing starter.** A's real dataset and B's
optimisations are integration inputs, not prerequisites for anyone else's implementation.
Assign actual names to A-D. Each member owns a component and report section and reviews
the assigned teammate's work weekly.

## Starting point available to everyone

Use the existing `DenseMatrix`, `SearchIndex`, `SearchResult`, CVEC format, scalar index,
`cvs::squared_l2`, and `cvs::nearer` as the initial contract. See [contracts.md](contracts.md).
Keep these public interfaces compatible while implementing new routes in separate files.
Everyone can generate the same development dataset locally without waiting for A:

```sh
python3 scripts/prepare_data.py --synthetic \
  --database-size 1000 --tuning-size 20 --test-size 20 \
  --dimensions 50 --seed 5206 --output data/processed/dev-seed5206
```

For ties, duplicate vectors, empty lists, and hand-checkable rankings, create tiny in-memory
fixtures in each component's tests. Development fixtures support implementation; final
performance conclusions use the proposal's real dataset and frozen evaluation settings.

| Member | Independent work starting day 1 | First deliverable | Later integration input | Report section | Reviews |
| --- | --- | --- | --- | --- | --- |
| A: data and correctness | Verify GloVe release, prepare real splits, extend FP64 validation, implement subset ground truth using the existing scalar index | Versioned preparation procedure, checksummed split manifest, reference-result format and oracle fixtures | Validate B/C results when their routes arrive; D consumes A's ground truth and manifest | Data, validation, limitations | C |
| B: CPU optimisation | Implement a separate optimised flat route on synthetic/in-memory data; work on blocked layout and SIMD behind the existing search interface | Optimised-flat route with scalar fallback and correctness/layout tests | Use A's real data for final measurements; expose a reusable scoring backend for C at the kernel checkpoint | Exhaustive-scan optimisations | D |
| C: IVF | Build assignments, inverted lists, centroid scoring/probing and top-k on synthetic data using existing scalar distance/order helpers; start list tests with fixed centroids | Scalar-backed IVF route; all-lists agreement with the existing scalar flat index | Add seeded training on development database vectors; later plug in B's scoring backend and A's real data | IVF design and recall/latency tradeoff | A |
| D: benchmarking and parallelism | Extend the harness using the existing scalar index; build recall checks with hand-written neighbour IDs, timing/raw-output schema, memory measurement and query threading | Generic runner accepting a `SearchIndex`, validated recall calculation, raw outputs and scalar batch scaling | Register B/C routes when ready; use A's ground truth/manifest for final runs | Setup, results, threading | B |

There is no implementation chain A -> B -> C -> D. A supplies final data, B supplies an
optional faster scoring backend during development, C supplies an additional search route,
and D supplies evaluation infrastructure. They converge at explicit integration checkpoints.

## Ownership and handoffs

| Owner | Primary files to add or extend | Handoff to the others |
| --- | --- | --- |
| A | `scripts/prepare_data.py`, a new ground-truth/subset script, data/oracle tests | CVEC files and manifest under the agreed format; exact neighbour IDs for each subset/query split |
| B | New `include/cvs/optimized_flat.hpp`, `src/optimized_flat.cpp`, layout/kernel tests | A `SearchIndex` implementation; later a documented candidate-scoring backend with a scalar fallback |
| C | New `include/cvs/ivf.hpp`, `src/ivf.cpp`, training script and IVF tests | A `SearchIndex` implementation with list/probe/seed metadata; explicit original-ID mappings |
| D | `apps/benchmark.cpp`, new benchmark/threading helpers, result/plot scripts and harness tests | Runner, per-query/result schema, recall/throughput/memory measurements |

New file paths above are work assignments, not features already implemented. The shared
headers remain the baseline contract. Agree on narrow changes in a short integration PR;
continue local work against the current interface meanwhile. Each component PR adds its
own CMake target/source entries. Keep these edits small so build-file conflicts are easy
to resolve. Cross-review runs alongside implementation and does not require another
member to finish their whole component first.

## Weeks 1-2: four independent implementations, then integration

- Day 1: everyone runs the quick start, reads the existing contracts and begins their row
  in the table. Review the contracts in a short kickoff; no member waits for another's code.
- First checkpoint: A demonstrates reproducible splits/reference fixtures; B demonstrates
  its flat route; C demonstrates list/probe correctness with fixed centroids; D demonstrates
  the runner/recall checks on the scalar route. Each demonstration can use development data.
- C adds seeded centroid training using synthetic database vectors while A prepares GloVe.
  Neither fixed test centroids nor synthetic training depend on A's real-data delivery.
- Initial integration: D registers any available route; A runs cross-route checks. C's
  all-lists search must match `ScalarFlatIndex` using the same scalar distance arithmetic,
  including ties and duplicates. C does not wait for B's optimised kernel for this check.
- A delivers the real 100k/500/1000 splits and manifests as soon as ready. Replace the
  development input paths without changing any route's query interface.

## Weeks 3-4: optimise and scale

- B measures layout and SIMD separately. C continues training/probing work with the scalar
  backend until B's reusable backend is ready; D develops threading on the scalar route.
- Kernel checkpoint: B/C agree on candidate IDs, scoring arithmetic, dimensions/padding,
  and deterministic selection. Retain original database IDs through blocked storage and
  lists. Both routes must select the same scoring backend for equivalence tests; C must
  not copy B's SIMD implementation into a separate version.
- A re-runs all-lists equivalence for each integrated backend. SIMD and scalar distances
  may differ through rounding; compare IVF to exhaustive search using the same backend.
- C tunes probes on the real tuning split across three training seeds once data and routes
  pass correctness. D adds representative 1/2/4-thread runs, capped by hardware, as each
  immutable route becomes available. A checks provenance and per-subset ground truth.
- At week 3, D helps whichever of B/C is behind after delivering the harness baseline.
  Reduce optimisation combinations if needed and retain the tested scalar path.

## Weeks 5-6: freeze, evaluate, explain

- A gates the freeze: record settings before using held-out test queries.
- Recompute exhaustive neighbours for each 10k/50k/100k database subset.
- Run frozen comparisons and produce raw outputs, recall/latency plots, scaling, build time,
  candidate counts, and peak-memory analysis.
- Explain overheads if IVF is slower. Preserve held-out evaluation even if scope is reduced.
- Assemble the report and demonstration; every member contributes.

These final steps have necessary dependencies: real-data measurements need A's dataset;
the final IVF-versus-optimised comparison needs both B and C; held-out evaluation needs
the tuning freeze. Those dependencies gate final results, not starting development.

## Contribution workflow

Work in small feature branches and open a pull request for the weekly cross-review.
Include the relevant correctness results and a reproducible run command. Keep datasets,
download archives, build files, and raw local runs out of Git. Curate final raw results and
plots into a versioned submission directory or release deliberately; ignored local output
folders alone are not a reproducibility archive. Do not add a software license on behalf
of the team without agreeing on one.
