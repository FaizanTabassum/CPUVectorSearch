# Locking the settings before final evaluation

The shared record lives at **`configs/evaluation-freeze.json`**.
It currently says **`"status": "pending"`**. This is a planning record, not a completed
freeze, and the team should not run the 1,000 final inputs yet. The benchmark CLI does
not automatically enforce this record; the team must observe the gate.

## What A needs from the team

1. C supplies the practice results and proposed IVF settings for each collection size:
   lists, probes, training seed(s), and training iterations. Starting defaults are not
   frozen decisions. Compare the agreed list/probe grid and all three training seeds.
2. D supplies the final measurement procedure: machine, thread counts, warmups,
   repetitions, memory method, and where raw measurements will be stored.
3. A checks the dataset hashes, per-size reference answers, and same-backend all-lists
   IVF agreement. A/B/C agree which distance calculation is used for each comparison.

## Completing the record

After tuning is complete, fill in the selected configurations, resolved scoring backend,
code commit, reference manifests, correctness evidence, tuning evidence, and timing
procedure. Keep the dataset identity pinned; the expected dataset-manifest checksum is
already recorded. Include evidence if no tested IVF setting meets the target, rather
than inventing a successful setting.

Then set `status` to `frozen`, record the actual UTC freeze time, and **commit the record
before any held-out search run**. Record the tested implementation commit in
`frozen_code_commit`; the later freeze-record commit provides the audit trail.
The existing `configs/experiments.json` is the planned search grid, not the completed
freeze record.

Only after that commit should the team generate backend-matched final reference answers
and run final evaluation. If settings need to change after seeing final results, preserve
the original freeze/results and document the deviation. Do not quietly overwrite the
record or present a retuned result as the original held-out evaluation.

## Files for Aravind's runner

After running `python3 scripts/restore_prepared_data.py`, the original practice answers are:

```text
data/processed/glove-50d-seed5206/reference-tuning/top10-tuning-n10000.csv
data/processed/glove-50d-seed5206/reference-tuning/top10-tuning-n50000.csv
data/processed/glove-50d-seed5206/reference-tuning/top10-tuning-n100000.csv
```

These use **scalar** scoring, 500 practice queries and ten neighbours per query.
`reference-manifest.json` in the same directory records input/output hashes. Regenerate
the exhaustive reference with the same scoring backend before an exact comparison that
uses different floating-point arithmetic. Restoring files does not run the final queries.
