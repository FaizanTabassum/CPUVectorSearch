# Team plan and acceptance milestones

Assign actual names to A-D when the team starts. Each member owns a component and a report
section, and reviews the assigned teammate's work weekly.

| Member | Owns | Starter coverage | Next implementation | Report section | Reviews |
| --- | --- | --- | --- | --- | --- |
| A | Data and validation | Seeded splits, format, small FP64 oracle | Verify exact GloVe release; extend validation and all-lists IVF check; ground-truth/subset runner; freeze tuning | Data, validation, limitations | C |
| B | Exhaustive scan | Scalar kernel and bounded-heap top-k | Blocked storage, shared optimised kernel, explicit SIMD; layout/SIMD ablations | Exhaustive-scan optimisations | D |
| C | IVF index | Shared search/result contract | Train 64/256 centroids on database only; assignments, inverted lists, probing; tune probes and record candidates | IVF design and recall/latency tradeoff | A |
| D | Parallelism and benchmarking | Single-thread smoke JSON and CI | Recall evaluator, final runner, memory measurement, 1/2/4-thread query batches, raw outputs and plots | Setup, results, threading | B |

## Weeks 1-2: establish correctness and integration

- Everyone runs the synthetic quick start and agrees on contracts.
- A prepares the real 100k/500/1000 splits with checksums and checks the source release.
- B preserves scalar correctness and exposes a shared kernel for future routes.
- C produces a basic index with valid assignments and list probing.
- D integrates real-file runs and independently checks recall computation on small fixtures.
- Acceptance: all-lists IVF matches the shared exhaustive kernel, including ties.

## Weeks 3-4: optimise and scale

- B introduces layout and SIMD independently so their effects can be measured.
- C tunes probe counts on tuning queries across three training seeds.
- D measures representative batches at 1/2/4 threads, capped by hardware.
- A extends regression coverage and checks split/ground-truth provenance.
- At week 3, D helps whichever of B/C is behind. Reduce optimisation combinations if needed.

## Weeks 5-6: freeze, evaluate, explain

- A gates the freeze: record settings before using held-out test queries.
- Recompute exhaustive neighbours for each 10k/50k/100k database subset.
- Run frozen comparisons and produce raw outputs, recall/latency plots, scaling, build time,
  candidate counts, and peak-memory analysis.
- Explain overheads if IVF is slower. Preserve held-out evaluation even if scope is reduced.
- Assemble the report and demonstration; every member contributes.

## Contribution workflow

Work in small feature branches and open a pull request for the weekly cross-review.
Include the relevant correctness results and a reproducible run command. Keep datasets,
download archives, build files, and raw local runs out of Git. Curate final raw results and
plots into a versioned submission directory or release deliberately; ignored local output
folders alone are not a reproducibility archive. Do not add a software license on behalf
of the team without agreeing on one.
