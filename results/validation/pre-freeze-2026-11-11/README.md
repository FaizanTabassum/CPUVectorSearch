# Member A: pre-freeze validation, 11 November 2026

`validation.json` records the checks and exact reproduction commands. All searches use
only the **500 tuning queries**. No search used the 1,000 held-out test queries.

- Dataset/archive checksums match the pinned GloVe snapshot. All three splits have the
  expected sizes, finite unit-normalised vectors and disjoint source rows.
- The archived tuning CSV covers all 144 IVF configurations plus six exhaustive baselines.
  The dataset and per-prefix reference hashes match Member A's validated files.
- All 13 Release test suites pass on macOS/ARM, including the independent FP64 oracle,
  deterministic k-means, shared-backend IVF, benchmark driver and reference tools.
- Exhaustive scalar scoring reproduces the original 10k/50k/100k reference IDs and FP32 scores.
- IVF with 256/256 lists probed matches those IDs and FP32 scores for all three sizes and
  all three seeds: nine exact, same-backend all-lists comparisons.
- IVF with the chosen 256/64 setting independently reproduces all nine recall values
  archived from Aravind's x86/AVX2 tuning sweep. Every seed is above 0.95.

| Database rows | Mean recall, selected setting | Worst seed |
| --- | --- | --- |
| 10,000 | 0.9642667 | 0.9632 |
| 50,000 | 0.9792667 | 0.9774 |
| 100,000 | 0.9827333 | 0.9820 |

The validated implementation is `f0185ef104d3cf74e6744aa2fd0d587048c0a9df`; merging that
branch with current main adds no benchmark source differences. Binary/source hashes and
commands in the record identify the evidence. Reproduce each listed command and compare
its returned neighbour IDs and FP32 scores with the exhaustive output for that prefix.

Native checks used the scalar backend because this machine is ARM. They establish
correctness and recall agreement, not final AVX2 performance. The archived x86 sweep's
manifest marks its tree dirty and does not include individual raw process files. Its
summary is retained as tuning evidence with that limitation; the final frozen run must
archive raw process/query files. The final machine and policy remain Aravind's Windows/
Intel/AVX2 procedure, with 5 timed passes, 3 batch passes and 3 independent processes.
