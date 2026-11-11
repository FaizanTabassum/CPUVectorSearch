# Report sections by Member D: Experimental setup, results and multi-threading

The tuning results are taken from `results/tuning/summary.md`. A total of 500 queries were used
for tuning, with each configuration running in a separate process and completing three timed
passes. Section 5.4 will be completed once the final evaluation results are available.

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

The complete procedure, including the number of repetitions and the locations of the raw
results, is documented in `docs/benchmarking.md`.

### 4.6 Answer sheets and scoring backend

A's reference answers were generated using scalar scoring, while the faster search routes use
AVX2. Since AVX2 may add floating-point values in a different order, we checked whether this
difference affected the final search results.

To do this, we generated another set of reference answers using an exhaustive AVX2 scan through
`scripts/make_references.py`. We then compared these answers with A's original reference sheets.

Across all three database sizes and the 500 tuning queries, both methods returned exactly the
same top-10 results. This confirms that using AVX2 does not change any of the recall values
reported in this experiment.

### 4.7 Tuning and final evaluation

The IVF settings were selected using only the 500 tuning queries. At each database size, we
chose the fastest configuration that achieved an average Recall@10 of at least 0.95 across the
three training seeds.

The selected settings are saved in `configs/evaluation-freeze.json` before the test queries are
run. The `run_benchmarks.py` script prevents the test set from being used until the freeze record
is marked as frozen, and it stores the hash of the freeze record alongside the final results.

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

### 5.2 Exhaustive search: scalar vs optimised

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

### 5.4 Held-out test results

*To be completed after the evaluation freeze, using the results stored in `results/final/`.*

This section will report the frozen IVF configurations and their Recall@10, median and p95
latency, build time and peak memory for all three database sizes: 10,000, 50,000 and 100,000
vectors.

It will also check whether IVF continues to achieve the target recall of 0.95 on the 1,000 test
queries that were not used during tuning.

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

*Source: `vector_search --route optimized_flat --threads 1,2,4` on 100,000 vectors and the 500
tuning queries. The tuning sweep in `results/tuning/` used one thread only, so it has no
thread-scaling figure.*

We measured how performance changed when increasing the number of threads for `optimized_flat`
with 100,000 vectors.

| Number of threads | Throughput (queries/sec) | Speed-up |
| ---: | ---: | ---: |
| 1 | 312 | 1.00× |
| 2 | 558 | 1.79× |
| 4 | 1,058 | 3.39× |

The results show a clear improvement as more threads are added. With four threads, throughput
increased from 312 to 1,058 queries per second, giving a speed-up of 3.39×. This corresponds to
approximately 85% parallel efficiency.

The workload was also distributed fairly evenly across the four threads, with each processing
between 120 and 129 queries.

However, the speed-up was not perfectly linear. One reason is that every query needs to read the
entire database, so multiple threads compete for memory bandwidth and shared cache resources.
The laptop may also reduce its processor clock speed when more cores are active, which limits
the overall improvement.

### 6.3 Limitations

There are a few limitations to consider when looking at these results:

- **Single-machine testing:** All experiments were performed on one laptop, so performance may
  differ on machines with different processors, memory bandwidth and power settings.
- **Throughput vs latency:** Multi-threading improves the number of queries processed per second,
  but it does not make an individual query finish faster.
- **Memory measurement:** Peak memory represents the usage of the entire benchmark process. On
  Windows, this value cannot be reset after loading the data, so it also includes memory used
  while reading the input files.
- **Tuning repetitions:** The tuning experiments used one process and three timed passes per
  configuration. The final evaluation follows the complete procedure described in
  `docs/benchmarking.md`.

---

## Overall summary

The experiments show that each search method has its own advantages. The scalar implementation
provides a simple baseline, while AVX2 improves exhaustive search performance by around 2.2×
without affecting recall.

IVF provides a further improvement by searching only selected parts of the database. It achieves
the target recall of 0.95 while examining roughly a quarter of the vectors, making it faster and
more memory-efficient than the optimised exhaustive search in the tested configurations.

Multi-threading also improves overall throughput, reaching a 3.39× speed-up with four threads.
However, memory bandwidth and processor limitations prevent the performance from scaling
perfectly with the number of threads.

Overall, the results highlight the trade-off between search accuracy, execution time, memory
usage and parallel performance.
