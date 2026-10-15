# Shared team dataset

These are the prepared files for our project. They are included in Git so every teammate
can start with the same data after cloning; no separate dataset download is needed.
Together with metadata and practice answer sheets, the snapshot is approximately 21 MiB.

| File | Contents |
| --- | --- |
| `database.cvec` | 100,000 searchable vectors, each containing 50 FP32 numbers |
| `tuning.cvec` | 500 separate practice inputs for choosing search settings |
| `test.cvec` | 1,000 separate final inputs; reserve these until settings are frozen |
| `manifest.json` | Selection seed, source row IDs, input/output checksums and preparation version |
| `validation.json` | Preparation checks, source/archive checksums and vector normalisation checks |
| `reference-tuning/` | Scalar top-ten practice answers for the 10k, 50k and 100k database prefixes |
| `SHA256SUMS` | Checksums for the snapshot files, including this document |

## Source and preparation

Source: Stanford **GloVe 2024 Wikipedia + Gigaword 5, 50-dimensional vectors**.
See the [GloVe project page](https://nlp.stanford.edu/projects/glove/) and
[original download](https://nlp.stanford.edu/data/wordvecs/glove.2024.wikigiga.50d.zip).
Stanford makes the pre-trained vectors available under the
[Public Domain Dedication and License v1.0](https://opendatacommons.org/licenses/pddl/1-0/).
This data attribution does not set the license for the project's source code.

The preparation script selected 101,500 rows from 1,291,147 source rows with seed **5206**,
shuffled them, and divided them into database, practice, and final sets. Source row IDs
do not overlap between the sets; different source rows may still have identical vectors.
Vectors were L2-normalised before being stored as FP32 values. The existing manifest
records Python 3.14.4 and the checksums needed to identify this exact snapshot.

The larger original ZIP and text file are not copied into Git. To repeat preparation,
follow the [setup guide](../../../docs/getting-started.md) and choose a new output directory.
Compare the resulting vector-file checksums with `manifest.json`.

## Verify the files

From this dataset directory on macOS:

```sh
shasum -a 256 -c SHA256SUMS
```

On Linux, `sha256sum -c SHA256SUMS` performs the same check.
All listed files should report `OK`.

## Using the practice answers

The CSV files contain ten neighbours per input: query row, rank, database row, and squared
distance. Row IDs start at zero; ranks start at one. Smaller distance means closer.
Each file has 500 inputs and 5,000 answer rows. The subset sizes are prefixes of the same
shuffled database, with exact answers recalculated for each size.

These are **scalar** reference answers. Their manifest preserves the original executable,
source fingerprints and build information from generation. Later source/build changes
do not rewrite that history. If a scoring backend changes floating-point arithmetic,
generate exhaustive answers with that backend before its final comparison.
See the [reference answer guide](../../../docs/reference-answers.md) for details.

Treat this directory as a fixed snapshot. Put new datasets, regenerated answers and
experiment outputs in separate ignored directories. No final-test answer sheets or
held-out search results are included.
