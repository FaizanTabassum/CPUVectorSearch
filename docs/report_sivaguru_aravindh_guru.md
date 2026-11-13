# Report sections by Member D: Experimental setup, results and multi-threading

Sections 5.2 and 5.3 use the tuning run in `results/tuning/` (500 tuning queries). Section 5.4
and Section 6.2 use the final run on the 1,000 held-out test queries, stored in
`results/final/frozen-2026-11-11/`.

---

## 4. Experimental setup

### 4.1 Machine and build

All experiments were carried out on the same laptop to make sure that each search method was
tested under identical conditions. The machine uses an Intel Core i7-10750H processor with
6 cores and 12 hardware threads, running Windows. We used GCC 16.2 through MSYS2 to compile the
code.

The project was built using CMake in Release mode with `-O3` optimisation. Floating-point
contraction was disabled using `-ffp-contract=off`, and fast-math was not enabled. This helps
keep the scalar distance calculations consistent. B's AVX2 implementation was compiled
separately and selected at runtime only if the processor supported it.

During benchmarking, the laptop was connected to power and set to its best-performance mode.
Other applications were closed to reduce interference with the measurements.

### 4.2 Data and queries

For the experiments, we used A's prepared GloVe 2024 Wikipedia and Gigaword word vectors. Each
vector has 50 dimensions, is L2-normalised, and is stored in FP32 format.

The full database contains 100,000 vectors. We also tested smaller databases containing 10,000
and 50,000 vectors. These were created by taking prefixes of the full dataset, which means that
vector IDs remain consistent across all three sizes.

We used 500 queries to select the best settings. A separate set of 1,000 queries was kept aside
for the final evaluation and was not used during tuning. Exact top-10 results for each database
size were generated using an exhaustive scalar search, as explained in Section 3.

### 4.3 Search routes

We compared three search approaches. Each one uses the same `SearchIndex::search(query, k)`
interface, with k set to 10. This allows all methods to be tested using the same benchmark
runner.

| Route | Description | Settings |
| --- | --- | --- |
| `scalar_flat` | Performs an exhaustive search using the scalar FP32 kernel. | None |
| `optimized_flat` | Performs the same exhaustive search using B's padded database copy and AVX2 kernel. | AVX2 selected at runtime (supported by this CPU). |
| `ivf` | Uses C's inverted file index, with centroids trained using seeded k-means. | 64 or 256 lists; probes from 1 to the total number of lists; seeds 5206, 5207 and 5208. |

Each configuration was run in a separate process, and the same benchmark runner was used to
measure all three approaches.

### 4.4 Metrics

We used the following metrics to compare the search methods:

- **Recall@10:** Measures how many of the returned top-10 results also appear in the exact top-10
  results. The score is averaged across all queries.
- **Latency:** Measures the time taken by a single `search()` call on one thread. We report the
  median, p95 and p99 values using the nearest-rank method. This includes scoring, top-k selection
  and result creation. For IVF, it also includes centroid scoring and collecting vectors from the
  selected lists.
- **Batch throughput:** Measures how many queries can be completed per second when processing the
  full query set. This is tested using 1, 2 and 4 threads, without individual query timers.
- **Candidates per query:** Counts the number of database vectors actually scored for each query.
- **Build time:** Measures how long it takes to construct the index, including k-means training
  for IVF.
- **Peak memory:** Records the highest resident memory used by the benchmark process, measured
  using Windows `PeakWorkingSetSize`.

### 4.5 Benchmark procedure

For every configuration, the runner first loads the input data and builds the index. It then
performs 10 warmup queries without recording their times.

After warmup, the full query set is searched several times using one thread. Latency is recorded
for these runs. Batch throughput is measured separately using 1, 2 and 4 threads, and the median
pass time is reported.

In the final run, each configuration used 10 warmup queries, 5 timed passes and 3 batch passes
at each thread count, and the whole process was repeated 3 times. We report the median over
the three processes, and for IVF we then take the mean over the three training seeds. The tuning
run was shorter: one process and three timed passes per configuration.

The complete procedure, including the locations of the raw results, is documented in
`docs/benchmarking.md`.

### 4.6 Answer sheets and scoring backend

A's reference answers were generated using scalar scoring, while the faster search routes use
AVX2. Since AVX2 may add floating-point values in a different order, we checked whether this
difference affected the final search results.

To do this, we generated another set of reference answers using an exhaustive AVX2 scan through
`scripts/make_references.py`. We then compared these answers with A's original reference sheets.

Across all three database sizes and the 500 tuning queries, both methods returned exactly the
same top-10 results. This confirms that using AVX2 does not change any of the recall values
reported for the tuning run.

For the final evaluation we did not assume this would also hold for the new queries. We made two
separate sets of answer sheets for the test queries, one with scalar scoring and one with AVX2.
`scalar_flat` was scored against the scalar sheets, and `optimized_flat` and `ivf` against the
AVX2 sheets, so each route is compared with answers computed the same way.

### 4.7 Tuning and final evaluation

The IVF settings were selected using only the 500 tuning queries. Our starting rule was to pick,
at each database size, the fastest configuration with an average Recall@10 of at least 0.95
across the three training seeds (Section 5.3).

When the team froze the settings, we chose one setting for all three sizes instead: 256 lists
and 64 probes, with 25 k-means iterations and all three seeds kept. This followed Devam's
recommendation from his own tuning sweep. The fastest setting at 50,000 and 100,000 vectors,
64 lists with 16 probes, does not reach 0.95 at 10,000 vectors and only just passes at 50,000
(0.954), so it leaves almost no margin for new queries. The 256/64 setting had a tuning recall of
0.964, 0.979 and 0.983 and was only about 7% and 3% slower than 64/16 at 50,000 and 100,000
vectors.

The frozen settings, the timing procedure and the code commit are saved in
`configs/evaluation-freeze.json`, which was committed on 11 November 2026 before any test query
was used. The `run_benchmarks.py` script refuses to run the test set until this record is marked
as frozen. No settings were changed after the test results were seen.

This separation helps ensure that the final test results are not influenced by the tuning
process.

---

## 5. Results

### 5.1 Checking that the measurements are correct

Before comparing the search methods, we first checked whether the measurement code was working
correctly. This was important because incorrect recall or timing calculations could affect the
conclusions of the experiment.

We tested the recall calculation using manually prepared neighbour lists. For example, when 9 out
of 10 returned IDs matched the reference results, the calculated recall was 0.9. We also checked
that repeated IDs were counted only once.

The scalar search returned exactly the same neighbours as A's reference answer sheet. We then
deliberately replaced two of the ten reference IDs for each query and confirmed that recall
dropped to exactly 0.8. This showed that the recall calculation was responding correctly to
changes in the results.

We also tested IVF by increasing the number of probes until every list was searched. In this
case, IVF returned exactly the same results as the exhaustive search.

### 5.2 Exhaustive search: scalar vs optimised (tuning queries)

*Figure: `results/tuning/latency_by_size.png`.*

We first compared the normal scalar search with the optimised AVX2 version. Both methods search
through every vector, but the optimised version uses SIMD instructions to process multiple values
more efficiently.

At 100,000 vectors, the scalar search took 6.20 ms per query, while the optimised version took
only 2.86 ms. This represents a speed-up of approximately 2.2×, with both methods maintaining a
recall of 1.0.

A similar improvement was observed at smaller database sizes:

| Database size | Scalar latency | Optimised latency | Speed-up |
| ---: | ---: | ---: | ---: |
| 10,000 | 0.57 ms | 0.25 ms | 2.3× |
| 50,000 | 2.90 ms | 1.33 ms | 2.2× |
| 100,000 | 6.20 ms | 2.86 ms | 2.2× |

As expected, latency increases as the database grows because both methods still need to examine
every vector.

The optimised version does require additional memory because it keeps a padded copy of the
database. At 100,000 vectors, its peak memory usage was 46.2 MiB, compared with 24.8 MiB for the
scalar version. Building the index added approximately 21.4 MiB.

### 5.3 IVF: recall vs latency on the tuning queries

*Figure: `results/tuning/recall_vs_latency.png`. Table: IVF sweep in `results/tuning/summary.md`.*

The next experiment looked at how the number of IVF probes affects recall and search time. In
general, increasing the number of probes improves recall because more database vectors are
considered. However, it also increases the time needed to answer each query.

For example, with 64 lists and 100,000 vectors, recall increased from 0.460 with one probe to
0.884 with eight probes. It reached 0.958 with 16 probes and 0.994 with 32 probes.

We selected the fastest configurations that achieved the target Recall@10 of at least 0.95.

| Database size | IVF configuration | Recall@10 | Median latency | Optimised flat latency | Vectors scored |
| ---: | --- | ---: | ---: | ---: | ---: |
| 10,000 | 256 lists, 64 probes | 0.964 | 0.146 ms | 0.247 ms | 2,552 |
| 50,000 | 64 lists, 16 probes | 0.954 | 0.702 ms | 1.333 ms | 12,612 |
| 100,000 | 64 lists, 16 probes | 0.958 | 1.912 ms | 2.857 ms | 25,269 |

These results show that IVF can achieve the required recall while examining only around a
quarter of the database vectors. It was also faster than the optimised exhaustive search at all
three database sizes. The speed-up was approximately:

- 1.7× at 10,000 vectors,
- 1.9× at 50,000 vectors,
- 1.5× at 100,000 vectors.

In single-thread batch testing with 100,000 vectors, IVF processed 503 queries per second,
compared with 335 queries per second for the optimised exhaustive search.

IVF also used less memory. At 100,000 vectors, its peak memory usage was 26.4 MiB, while the
optimised exhaustive search required 46.2 MiB. This is mainly because IVF does not need to
maintain a separate padded copy of the entire database.

We also compared different IVF configurations using 256 lists. At 100,000 vectors, 32 probes
achieved a recall of 0.942 in 0.99 ms, which was close to the target but still below 0.95.
Increasing the number of probes to 64 raised recall to 0.983, but latency increased to 1.98 ms.
This was slightly slower than using 64 lists with 16 probes.

At 10,000 vectors, however, 64 lists required 32 probes to reach a recall of 0.985 in 0.23 ms.
In this case, using 256 lists with 64 probes gave a better result.

Although IVF examines far fewer vectors, the reduction in search time is not as large as the
reduction in candidates. This is because IVF also needs to calculate distances to the centroids,
collect vectors from different memory locations and process shorter groups of vectors using
AVX2.

Another trade-off is the time needed to build the index. For example, training the IVF index
with 64 lists at 100,000 vectors took 9.6 seconds, while building the optimised flat index took
only 9 ms.

Finally, we checked whether the results were consistent across the three training seeds. The
recall difference between seeds was no more than 0.03 for any configuration. For the three
selected configurations, the difference was even smaller, at no more than 0.003.

As explained in Section 4.7, the setting finally frozen was 256 lists with 64 probes at every
size, which gives up a little speed at 50,000 and 100,000 vectors for a larger recall margin.

### 5.4 Held-out test results

*Figures: `results/final/frozen-2026-11-11/combined/report/`. Tables:
`results/final/frozen-2026-11-11/combined/report/summary.md`.*

The final run used the frozen settings on the 1,000 test queries, which were not used at any
point during tuning. It was run on the same laptop as before: 9 scalar runs and 36 AVX2 runs
(optimised flat, and IVF with three seeds, each repeated in three separate processes).

**Recall.** IVF stayed above the 0.95 target at all three sizes, and the results were very close
to the tuning run:

| Database size | Tuning recall | Test recall (mean of seeds) | Worst seed on test |
| ---: | ---: | ---: | ---: |
| 10,000 | 0.964 | 0.965 | 0.962 |
| 50,000 | 0.979 | 0.978 | 0.977 |
| 100,000 | 0.983 | 0.981 | 0.979 |

Recall changed by no more than about 0.002 on the new queries, and even the worst seed was well above 0.95.
This shows the settings were not over-fitted to the tuning queries. Both flat routes had a recall
of exactly 1.0, as expected for an exhaustive search.

**Latency.** Single-thread latency per query (ms):

| Database size | Route | Recall@10 | Median | p95 | p99 | Vectors scored |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 10,000 | `scalar_flat` | 1.000 | 0.565 | 0.628 | 0.830 | 10,000 |
| 10,000 | `optimized_flat` | 1.000 | 0.246 | 0.266 | 0.418 | 10,000 |
| 10,000 | `ivf` 256/64 | 0.965 | 0.146 | 0.168 | 0.245 | 2,552 |
| 50,000 | `scalar_flat` | 1.000 | 2.841 | 3.170 | 4.183 | 50,000 |
| 50,000 | `optimized_flat` | 1.000 | 1.310 | 1.492 | 2.163 | 50,000 |
| 50,000 | `ivf` 256/64 | 0.978 | 0.710 | 0.838 | 1.319 | 12,551 |
| 100,000 | `scalar_flat` | 1.000 | 5.907 | 6.523 | 8.121 | 100,000 |
| 100,000 | `optimized_flat` | 1.000 | 2.832 | 3.097 | 4.364 | 100,000 |
| 100,000 | `ivf` 256/64 | 0.981 | 1.940 | 2.282 | 3.468 | 25,128 |

The optimised flat search was 2.3×, 2.2× and 2.1× faster than the scalar search at 10,000,
50,000 and 100,000 vectors. IVF was faster again: 1.7×, 1.8× and 1.5× faster than the optimised
flat search at the median, and 1.6×, 1.8× and 1.4× at p95. At 100,000 vectors, IVF was about
3.0× faster than the scalar baseline. IVF scored about 25% of the database at every size, which
matches searching 64 of the 256 lists.

**Build time.** IVF is the only route with a real build cost. Training 256 lists with k-means took
3.7 s at 10,000 vectors, 18.5 s at 50,000 and 36.9 s at 100,000, against 9 ms for the optimised
flat copy at 100,000. This is much more than the 9.6 s for 64 lists in the tuning run, because
every k-means iteration compares every vector with every centroid. The cost is paid once per
index, so it only pays off when many queries are searched.

**Memory.** At 100,000 vectors, the peak memory was 47.1 MiB for the optimised flat search,
27.4 MiB for IVF and 25.7 MiB for the scalar search. The increase in memory during the index build
shows the difference clearly: 21.4 MiB for the optimised flat search (its padded copy of the
database), 1.8 MiB for IVF (centroids and the vector IDs in each list) and almost nothing for the
scalar search. At 10,000 and 50,000 vectors all three routes had the same peak (26.1 and
33.8 MiB), because the peak happens while the full data file is loaded and the subset is copied,
before any index is built.

Overall, the held-out results confirm the tuning results: IVF with the frozen settings reached the
recall target on new queries and was the fastest route at every size, at the cost of a long
build.

---

## 6. Multi-threading

### 6.1 Approach

For multi-threading, we decided to divide the work across queries rather than splitting a single
query between multiple threads. Each thread takes the next available query from a shared atomic
counter and performs the search independently.

This approach avoids the need for locks because the `search()` function is `const` and each
thread maintains its own local scratch space. Assigning one query at a time also helps distribute
the workload evenly, especially for IVF, where different queries may search lists of different
sizes.

We also placed each thread's counters in separate cache lines to avoid false sharing, which can
reduce performance when multiple threads frequently update nearby memory locations.

To make the timing fair, all threads were created and kept ready before the benchmark started.
The timer stopped only after the last thread finished its assigned queries, so thread creation
time was not included.

After the benchmark, we ran the queries again using multiple threads and compared the results
with the single-thread output. All results matched exactly, bit for bit. We also tested the
threading implementation using ThreadSanitizer, which reported no data races.

### 6.2 Results

*Figure: `results/final/frozen-2026-11-11/combined/report/thread_scaling.png`. Source: the final
run on the 1,000 test queries (median of three processes, three batch passes each).*

We measured batch throughput with 1, 2 and 4 threads for every route. At 100,000 vectors:

| Route | 1 thread | 2 threads | 4 threads | Speed-up at 4 | Efficiency |
| --- | ---: | ---: | ---: | ---: | ---: |
| `scalar_flat` | 166 | 276 | 538 | 3.25× | 81% |
| `optimized_flat` | 346 | 536 | 1,007 | 2.92× | 73% |
| `ivf` 256/64 | 506 | 758 | 1,368 | 2.71× | 68% |

(Throughput in queries per second; efficiency is the speed-up divided by 4.)

All routes got faster with more threads, and IVF stayed the fastest at every thread count. With
four threads, IVF answered 1,368 queries per second, about 8 times the single-thread scalar
baseline. In all 45 runs, the multi-threaded results matched the single-thread results exactly.

The speed-up was not linear, and it was different for each route. At 10,000 vectors, where the
database is only about 2 MB and fits in the processor's 12 MB L3 cache, all three routes reached
about 3.0× with four threads. Since memory is not the limit there, the main reason is likely
that the laptop lowers its clock speed when more cores are busy. At 100,000 vectors (about
20 MB), the routes start to differ. The scalar search spends the most time on arithmetic per
vector, so it is least affected by threads sharing memory and scaled best. The AVX2 search reads
the same data much faster, so the four threads compete more for memory bandwidth. IVF scaled the
least: each query does less work, it reads vectors from scattered places in memory, and the
probed lists have different sizes, so some queries take longer than others.

### 6.3 Limitations

There are a few limitations to consider when looking at these results:

- **Single-machine testing:** All experiments were performed on one laptop, so performance may
  differ on machines with different processors, memory bandwidth and power settings.
- **Throughput vs latency:** Multi-threading improves the number of queries processed per second,
  but it does not make an individual query finish faster.
- **Memory measurement:** Peak memory represents the usage of the entire benchmark process. On
  Windows, this value cannot be reset after loading the data, so it also includes memory used
  while reading the input files.
- **Run-to-run variation:** The tuning experiments used one process and three timed passes per
  configuration, so their timings are less reliable than the final run. Laptop timings also
  vary between runs: an earlier single run of `optimized_flat` with four threads on the tuning
  queries reached 3.39×, compared with 2.92× in the final run. We report the final run, which
  takes the median of three separate processes.
- **IVF build time:** The frozen IVF setting needs up to 37 s of k-means training at 100,000
  vectors. This is not included in the query latency, but it matters if the index has to be
  rebuilt often.
- **One setting for all sizes:** We froze one IVF setting for all three sizes. A setting chosen
  for each size could be a little faster, but would leave less recall margin.

---

## Overall summary

The experiments show that each search method has its own advantages. The scalar implementation
provides a simple and exact baseline, and AVX2 makes the exhaustive search about 2.1 to 2.3 times
faster without changing the results.

IVF gives a further improvement by searching only part of the database. With the frozen setting
of 256 lists and 64 probes, it reached a Recall@10 of 0.965 to 0.981 on the 1,000 held-out test
queries, almost the same as on the tuning queries. It was 1.5 to 1.8 times faster than the
optimised exhaustive search and about 3 times faster than the scalar search at 100,000 vectors,
while using much less extra memory. The cost is a k-means build of up to 37 seconds.

Multi-threading across queries improves throughput for every route, with speed-ups of 2.7× to
3.3× on four threads at 100,000 vectors, and the results stay exactly the same as on one thread.
Clock speed limits and shared memory bandwidth stop the speed-up from being linear.

Overall, the results show the trade-off between accuracy, search time, build time, memory usage
and parallel performance.
