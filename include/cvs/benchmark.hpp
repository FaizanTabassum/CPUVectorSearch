#pragma once

// Member D: measurement helpers shared by the benchmark CLI and its tests.
//
// Nothing in here knows which search route it is timing. Everything goes
// through the SearchIndex interface, so B's and C's routes plug in without
// touching this file. Recall is always computed from real results, never
// filled in (contracts.md).

#include "cvs/search.hpp"

#include <chrono>
#include <cstddef>
#include <optional>
#include <string>
#include <vector>

namespace cvs::bench {

using Clock = std::chrono::steady_clock;

double milliseconds(Clock::time_point start, Clock::time_point end);

// Nearest-rank percentile, the convention in contracts.md. `sorted` must be
// ascending and non-empty; p is in (0, 1]. p = 0.5 gives the median rank.
double nearest_rank_percentile(const std::vector<double>& sorted, double p);

// ---------------------------------------------------------------- recall --

// Exact neighbour IDs per query, read from Member A's answer sheets
// (query_id,rank,neighbor_id,squared_distance). ids[q] is in rank order.
struct GroundTruth {
    std::size_t k = 0;                          // neighbours stored per query
    std::vector<std::vector<std::size_t>> ids;  // one row per query
};

// Reads and checks an answer sheet: header, contiguous query IDs from 0,
// ranks 1..k for every query, no repeated neighbour within a query.
GroundTruth load_ground_truth_csv(const std::string& path);

// |returned top-k IDs ∩ reference top-k IDs| / k. Only the first k entries
// of each side count. Returning fewer than k neighbours is simply penalised.
double recall_at_k(const std::vector<Neighbor>& found,
                   const std::vector<std::size_t>& expected, std::size_t k);
double recall_at_k(const std::vector<std::size_t>& found_ids,
                   const std::vector<std::size_t>& expected, std::size_t k);

struct RecallSummary {
    double mean = 0.0;
    double min = 0.0;
    std::size_t perfect_queries = 0;  // queries with recall exactly 1.0
    std::vector<double> per_query;
};

RecallSummary evaluate_recall(const std::vector<SearchResult>& results,
                              const GroundTruth& truth, std::size_t k);

// ------------------------------------------------------------ data prep --

// One std::vector<float> per row, ready to pass to SearchIndex::search.
std::vector<std::vector<float>> copy_rows(const DenseMatrix& matrix);

// First `rows` vectors as a new matrix. IDs keep their values because a
// prefix never renumbers anything.
DenseMatrix prefix_subset(const DenseMatrix& matrix, std::size_t rows);

// -------------------------------------------------------- single thread --

struct QueryRecord {
    std::size_t repetition = 0;
    std::size_t query_id = 0;
    double latency_ms = 0.0;
    std::size_t candidates_examined = 0;
};

struct LatencyRun {
    std::vector<QueryRecord> records;  // every timed search, in run order
    double mean_ms = 0.0;
    double median_ms = 0.0;
    double p95_ms = 0.0;
    double p99_ms = 0.0;
    double max_ms = 0.0;
    double mean_candidates = 0.0;
};

// Warms up on the first `warmup` queries, then times every query once per
// repetition on the calling thread. One clock pair wraps each search call.
LatencyRun measure_latency(const SearchIndex& index,
                           const std::vector<std::vector<float>>& queries,
                           std::size_t k, std::size_t warmup, std::size_t repetitions);

std::vector<std::size_t> cap_thread_counts(const std::vector<std::size_t>& requested,
                                           std::size_t hardware);

std::vector<SearchResult> search_all(const SearchIndex& index,
                                     const std::vector<std::vector<float>>& queries,
                                     std::size_t k, std::size_t threads);

struct BatchRun {
    std::size_t threads = 1;
    std::vector<double> wall_ms;
    double median_wall_ms = 0.0;
    double queries_per_second = 0.0;
    double best_queries_per_second = 0.0;
    std::vector<std::size_t> queries_per_thread;
};

BatchRun measure_batch(const SearchIndex& index,
                       const std::vector<std::vector<float>>& queries,
                       std::size_t k, std::size_t threads, std::size_t repetitions);

bool same_results(const std::vector<SearchResult>& a, const std::vector<SearchResult>& b);

struct MemorySnapshot {
    std::optional<std::size_t> current_rss_bytes;
    std::optional<std::size_t> peak_rss_bytes;
};

MemorySnapshot read_memory();
bool reset_peak_memory();
const char* memory_method();

std::string cpu_model();
std::size_t hardware_threads();

} // namespace cvs::bench
