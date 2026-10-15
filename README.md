# CPUVectorSearch

A CEG5206 team project about finding similar entries faster on a computer's processor (CPU).

## What is the idea?

Our dataset represents each word as a list of 50 numbers. This list is called a **vector**.
Given one such list, our program finds the **10 closest lists** in a collection of entries.
Closeness is calculated from the numbers; it does not always mean the words have similar meanings.

We are comparing three ways to search:

1. **Basic search:** check every entry, one by one.
2. **Faster full search:** still check every entry, but make better use of the processor.
3. **Grouped search (IVF):** organise entries into groups and search selected groups.
   This can reduce the work, but it may miss some close matches.

Our question: **can grouped search beat the faster full search while finding at least
95% of the reference matches across many searches?** This is a goal to investigate.

## What happens during a search?

1. Take one prepared list of numbers as the search input, also called a **query**.
2. Use one of the three methods to compare it with entries in the collection.
3. Return the ten closest entries, ordered from closest to furthest away.

Each returned entry has an identifying number and a **distance score**. A smaller score
means its numbers are closer to the input. The program calculates these results automatically.

## How do we compare the methods?

The basic search checks every entry and creates an answer sheet. We give the faster
methods the same inputs and compare their returned entries with those reference answers.
Finding nine of the ten reference entries means **90% recall** for that search. We average
this percentage across the inputs to measure how many reference matches a method finds.

We will test collections of **10,000, 50,000, and 100,000 entries** and measure search time,
match recovery, and memory use. We also compare running searches on one processor core
with spreading multiple searches across several cores.

## What is ready?

The basic exact search and B's faster exact search are implemented. The faster search
uses padded storage and AVX2 scoring when the processor supports it, with a scalar fallback.
B's ablation tool compares the basic search, the new storage layout with scalar scoring,
and the AVX2 route. Synthetic 50-dimensional benchmark results are recorded in
[docs/optimized-flat-results.md](docs/optimized-flat-results.md); real-data validation remains.

The repository also includes data-preparation tools, correctness checks using an independent
FP64 reference, a tool for saving exact reference answers, and a simple timing benchmark.
The grouped search (IVF) is implemented and can use the same scalar or AVX2 scorer as flat
search. Searches across multiple processor cores and the full comparison study are still to
be implemented.

## Getting started

Follow the [setup and first-run guide](docs/getting-started.md). You can try the program
with a small generated dataset before downloading the real data.

For the real study, we use GloVe 2024 Wikipedia + Gigaword, with 50 numbers per word:

- **100,000 entries** to search through.
- **500 practice inputs** to help choose the search settings.
- **1,000 final inputs** saved for evaluation after the settings are chosen.

Keeping final inputs separate helps us judge the chosen settings on fresh examples.
Keep downloaded data and generated answer sheets on your computer. The data folders
are already excluded from Git.

For B's layout/SIMD ablation, run the dedicated tool on prepared CVEC files:

```sh
./build/flat_ablation \
  --database data/processed/smoke/database.cvec \
  --queries data/processed/smoke/tuning.cvec \
  --k 10 --warmup-queries 5 --repetitions 5 --data-seed 5206 \
  --output results/raw/flat-ablation-smoke.json
```

The JSON records build time, padded index bytes, query latency and throughput, and top-k
ranking differences against the basic search. Synthetic timings check that the tool works;
they are not results on the real dataset.

## Work split

| Member | Main responsibility |
| --- | --- |
| A | Prepare reliable data and reference answers; check correctness |
| B | Make the full search faster |
| C | Build the grouped search |
| D | Measure speed, compare results, and run searches across multiple cores |

Everyone can start with the existing basic search and generated data. See the
[team plan](docs/team-plan.md) for independent starting tasks and integration checkpoints.

## More help

- [Setup, sample run, and dataset download](docs/getting-started.md)
- [Member A: creating and reading reference answers](docs/reference-answers.md)
- [B: optimized flat results](docs/optimized-flat-results.md)
- [Shared technical rules](docs/contracts.md)
- [Planned experiments](configs/experiments.json) (the full experiment runner is still to be built)
