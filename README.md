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

The starter has a working basic search, data-preparation tools, checks for mistakes,
and a program that saves the correct matches as **reference answer sheets**.
There is also a simple tool for measuring search time.

The faster full search, grouped search, work across multiple processor cores, and full
comparison study are the next development tasks for this starter.

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

## Who does what?

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
- [Shared technical rules](docs/contracts.md)
- [Planned experiments](configs/experiments.json) (the full experiment runner is still to be built)
