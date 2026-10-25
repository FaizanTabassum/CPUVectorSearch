# Setup and first run

Run the commands below from inside the `CPUVectorSearch` folder. This guide covers
macOS/Linux and keeps the commands out of the main project overview.

## 1. Check your tools

You need a C++17 compiler, CMake 3.16 or newer, and Python 3.9 or newer.
The Python scripts use the standard library, so no extra Python packages are required.
On macOS, Apple Command Line Tools provide a compiler; CMake is installed separately.
On Linux, a C++ compiler such as GCC and CMake are needed.

```sh
c++ --version
cmake --version
python3 --version
```

## 2. Build the program and run its checks

These commands prepare a runnable program in `build/` and run the correctness checks:

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel 2
ctest --test-dir build --output-on-failure
```

The expected outcome is that all tests pass. Stop and investigate failures before
using generated reference answers or making performance comparisons.

## 3. Try a small example

This creates made-up data so you can check that the program works without a download:

```sh
python3 scripts/prepare_data.py --synthetic \
  --database-size 1000 --tuning-size 20 --test-size 20 \
  --output data/processed/smoke

./build/vector_search \
  --database data/processed/smoke/database.cvec \
  --queries data/processed/smoke/tuning.cvec \
  --k 10 --output results/raw/smoke.json
```

Open `results/raw/smoke.json` to see the measurements. `--k 10` means return ten matches.
The timing results from this made-up dataset only show that the program runs; the real
project conclusions need the agreed dataset and completed comparison methods.

Preparation refuses to overwrite an existing dataset. If you have run this example
already, reuse its files or choose a different directory after `--output`.
The benchmark output file itself can be replaced when that benchmark is rerun.

## 4. Use local real data

To obtain the exact prepared dataset and practice answers used by Member A, run:

```sh
python3 scripts/restore_prepared_data.py
```

The command retrieves the pinned Git snapshot if needed and verifies all file checksums.
An identical existing dataset is verified and reused; different existing files are left
untouched with an error. Use `--output` with a new path for a separate copy.
The restored files stay local and ignored by Git. You can then use them directly:

```sh
./build/vector_search \
  --database data/processed/glove-50d-seed5206/database.cvec \
  --queries data/processed/glove-50d-seed5206/tuning.cvec \
  --k 10 --output results/raw/glove-scalar.json
```

Keep the test inputs reserved until settings are frozen. The record is
`configs/evaluation-freeze.json`, initially marked pending; see the
[freeze guide](evaluation-freeze.md). To reproduce selection from the original Stanford
source instead, follow the optional download and preparation steps below.

### Optional: download the original source

Use the 50-dimensional GloVe 2024 Wikipedia + Gigaword vectors listed on the
[official Stanford page](https://nlp.stanford.edu/projects/glove/). Check the source's
usage terms. The commands download the ZIP into the ignored local data folder and
extract the vector text file with a convenient name:

```sh
mkdir -p data/raw

curl -fL --retry 3 \
  https://nlp.stanford.edu/data/wordvecs/glove.2024.wikigiga.50d.zip \
  -o data/raw/glove.2024.wikigiga.50d.zip

unzip -p data/raw/glove.2024.wikigiga.50d.zip '*.txt' \
  > data/raw/glove.2024.wikigiga.50d.txt
```

If you already downloaded and extracted it, continue with your existing files.
The commands above replace files with those names if run again.

## 5. Prepare the dataset

After downloading and extracting the source, prepare the local dataset. Choose an output
directory that does not already contain data:

```sh
python3 scripts/prepare_data.py \
  --glove data/raw/glove.2024.wikigiga.50d.txt \
  --database-size 100000 --tuning-size 500 --test-size 1000 \
  --dimensions 50 --seed 5206 \
  --output data/processed/glove-50d-seed5206-regenerated
```

The script selects entries, puts their number lists on a common scale, and makes three
separate sets. `--seed 5206` lets the selection be repeated with the same source and
preparation procedure. The output directory must be new or empty.

| File | Purpose |
| --- | --- |
| `database.cvec` | 100,000 searchable entries |
| `tuning.cvec` | 500 practice inputs for choosing settings |
| `test.cvec` | 1,000 final inputs, reserved until settings are frozen |
| `manifest.json` | Records the selection and file checksums for reproducibility |

The script records the source rows so we can check that the sets do not overlap.
Different source rows can still contain identical vectors. The `.cvec` files store
numbers for the program to read; they are not ordinary text documents.

For Member A's next step, follow the [reference answer guide](reference-answers.md).
It generates separate answer sheets for 10k, 50k, and 100k database entries using the
practice inputs. The final-input workflow and full recall evaluation remain future work.

## Where files belong

| Folder | Contents |
| --- | --- |
| `apps/`, `src/`, `include/` | C++ programs and shared search code |
| `scripts/` | Data preparation and reference-answer commands |
| `tests/` | Checks for mistakes |
| `docs/` | Guides, team plan, and shared rules |
| `configs/` | Planned experiment settings; no full runner consumes them yet |
| `data/raw/` | Downloaded data, kept out of Git |
| `data/processed/` | Prepared datasets and new runs, kept local and ignored by Git |
| `results/raw/`, `results/plots/` | Local measurements and plots, kept out of Git |

The final submission still needs deliberately archived results, plots, source, tests,
report, and demonstration. Ignored output folders alone do not preserve a submission.

## Optional: extra memory checks

With Clang/GCC, use a separate build to check for invalid memory access and related errors:

```sh
cmake -S . -B build-sanitize -DCMAKE_BUILD_TYPE=Debug -DCVS_SANITIZERS=ON
cmake --build build-sanitize --parallel 2
ctest --test-dir build-sanitize --output-on-failure
```

For the numeric file format, tie handling, and measurement definitions, see the
[shared technical rules](contracts.md). All-lists IVF equivalence must use the same
scoring backend as its exhaustive reference. Quantisation is optional and outside the
starter. The project targets a CPU machine with 8 GB RAM, under 512 MB search-engine
memory, and approximately 3 GB disk for data and outputs; no GPU or paid service is needed.
