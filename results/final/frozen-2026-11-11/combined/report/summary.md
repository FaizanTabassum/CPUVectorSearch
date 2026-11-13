# Benchmark summary

Medians over process runs; IVF figures are averaged over training seeds.
IVF rows show the fastest setting with mean recall@10 >= 0.95.

## Single-thread latency, recall and build

| Rows | Route | Recall@10 | Median ms | p95 ms | p99 ms | Candidates | Build ms |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 10,000 | ivf (backend=avx2;lists=256;probes=64) | 0.9645 | 0.146 | 0.168 | 0.245 | 2,552 | 3,699 |
| 10,000 | optimized_flat (backend=avx2) | 1.0000 | 0.246 | 0.266 | 0.418 | 10,000 | 0.932 |
| 10,000 | scalar_flat | 1.0000 | 0.565 | 0.628 | 0.830 | 10,000 | 0.012 |
| 50,000 | ivf (backend=avx2;lists=256;probes=64) | 0.9778 | 0.710 | 0.838 | 1.319 | 12,551 | 18,460 |
| 50,000 | optimized_flat (backend=avx2) | 1.0000 | 1.310 | 1.492 | 2.163 | 50,000 | 4.488 |
| 50,000 | scalar_flat | 1.0000 | 2.841 | 3.170 | 4.183 | 50,000 | 0.012 |
| 100,000 | ivf (backend=avx2;lists=256;probes=64) | 0.9805 | 1.940 | 2.282 | 3.468 | 25,128 | 36,875 |
| 100,000 | optimized_flat (backend=avx2) | 1.0000 | 2.832 | 3.097 | 4.364 | 100,000 | 8.987 |
| 100,000 | scalar_flat | 1.0000 | 5.907 | 6.523 | 8.121 | 100,000 | 0.011 |

## Batch throughput (queries per second)

| Rows | Route | 1 thread | 2 threads | 4 threads | Best speedup | Same results |
| ---: | --- | ---: | ---: | ---: | ---: | --- |
| 10,000 | ivf (backend=avx2;lists=256;probes=64) | 6,710 | 10,869 | 20,206 | 2.99x | yes |
| 10,000 | optimized_flat (backend=avx2) | 3,943 | 6,357 | 11,871 | 3.01x | yes |
| 10,000 | scalar_flat | 1,725 | 2,773 | 5,329 | 3.10x | yes |
| 50,000 | ivf (backend=avx2;lists=256;probes=64) | 1,367 | 2,077 | 3,965 | 2.87x | yes |
| 50,000 | optimized_flat (backend=avx2) | 740 | 1,138 | 2,155 | 2.92x | yes |
| 50,000 | scalar_flat | 343 | 551 | 1,075 | 3.13x | yes |
| 100,000 | ivf (backend=avx2;lists=256;probes=64) | 506 | 758 | 1,368 | 2.71x | yes |
| 100,000 | optimized_flat (backend=avx2) | 346 | 536 | 1,007 | 2.92x | yes |
| 100,000 | scalar_flat | 166 | 276 | 538 | 3.25x | yes |

## Memory (MiB)

| Rows | Route | Peak RSS | RSS after build | Index RSS delta |
| ---: | --- | ---: | ---: | ---: |
| 10,000 | ivf (backend=avx2;lists=256;probes=64) | 26.1 | 8.0 | 0.25 |
| 10,000 | optimized_flat (backend=avx2) | 26.2 | 9.9 | 2.16 |
| 10,000 | scalar_flat | 26.1 | 7.8 | 0.02 |
| 50,000 | ivf (backend=avx2;lists=256;probes=64) | 33.8 | 16.4 | 0.97 |
| 50,000 | optimized_flat (backend=avx2) | 33.8 | 26.1 | 10.70 |
| 50,000 | scalar_flat | 33.8 | 15.4 | 0.02 |
| 100,000 | ivf (backend=avx2;lists=256;probes=64) | 27.4 | 26.7 | 1.77 |
| 100,000 | optimized_flat (backend=avx2) | 47.1 | 46.3 | 21.38 |
| 100,000 | scalar_flat | 25.7 | 24.9 | 0.02 |

## IVF sweep (all settings)

| Rows | Lists | Probes | Recall@10 (mean of seeds) | Seed min-max | Median ms | Candidates |
| ---: | ---: | ---: | ---: | --- | ---: | ---: |
| 10,000 | 256 | 64 | 0.9645 | 0.9617-0.9684 | 0.146 | 2,552 |
| 50,000 | 256 | 64 | 0.9778 | 0.9768-0.9785 | 0.710 | 12,551 |
| 100,000 | 256 | 64 | 0.9805 | 0.9794-0.9818 | 1.940 | 25,128 |
