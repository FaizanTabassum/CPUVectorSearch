

#include "cvs/benchmark.hpp"
#include "cvs/routes.hpp"

#include <atomic>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

int failures = 0;

void check(bool condition, const std::string& message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << '\n';
        ++failures;
    }
}

template <typename F>
void check_throws(F function, const std::string& message) {
    try {
        function();
    } catch (const std::exception&) {
        return;
    }
    check(false, message + " (expected an exception)");
}

bool close(double a, double b) { return std::fabs(a - b) < 1e-12; }

std::vector<cvs::Neighbor> neighbours(const std::vector<std::size_t>& ids) {
    std::vector<cvs::Neighbor> out;
    for (std::size_t i = 0; i < ids.size(); ++i) out.push_back({ids[i], static_cast<float>(i)});
    return out;
}

std::string write_temp(const std::string& name, const std::string& text) {
    const std::string path = "benchmark_tests_" + name + ".csv";
    std::ofstream(path) << text;
    return path;
}



class FirstHalfIndex final : public cvs::SearchIndex {
public:
    explicit FirstHalfIndex(const cvs::DenseMatrix& db)
        : half_(cvs::bench::prefix_subset(db, db.rows() / 2)), inner_(half_) {}
    std::size_t dimensions() const noexcept override { return half_.dimensions(); }
    cvs::SearchResult search(const std::vector<float>& q, std::size_t k) const override {
        return inner_.search(q, k);
    }
private:
    cvs::DenseMatrix half_;
    cvs::ScalarFlatIndex inner_;
};

cvs::DenseMatrix line_database(std::size_t rows) {
   
    std::vector<float> values;
    for (std::size_t i = 0; i < rows; ++i) {
        values.push_back(static_cast<float>(i));
        values.push_back(0.0F);
    }
    return cvs::DenseMatrix(2, values);
}


std::vector<cvs::SearchResult> search_each(const cvs::SearchIndex& index,
                                           const std::vector<std::vector<float>>& queries,
                                           std::size_t k) {
    std::vector<cvs::SearchResult> results;
    for (const auto& query : queries) results.push_back(index.search(query, k));
    return results;
}

// ------------------------------------------------------------------ tests

void test_recall_by_hand() {
    using cvs::bench::recall_at_k;
   
    check(close(recall_at_k(neighbours({3, 1, 2}), {1, 2, 3}, 3), 1.0), "reordered IDs give recall 1");
    check(close(recall_at_k(neighbours({1, 2, 9}), {1, 2, 3}, 3), 2.0 / 3.0), "two of three");
    check(close(recall_at_k(neighbours({7, 8, 9}), {1, 2, 3}, 3), 0.0), "none shared");
   
    check(close(recall_at_k(neighbours({1}), {1, 2, 3}, 3), 1.0 / 3.0), "short result");
    check(close(recall_at_k(neighbours({}), {1, 2, 3}, 3), 0.0), "empty result");
  
    check(close(recall_at_k(neighbours({1, 1, 1}), {1, 2, 3}, 3), 1.0 / 3.0), "duplicates once");
    
    check(close(recall_at_k(neighbours({4, 1}), {1, 2, 3, 4, 5}, 2), 0.5),
          "ID 4 is outside the reference top-2");
    check(close(recall_at_k(neighbours({1, 9, 2}), {1, 2, 3}, 2), 0.5),
          "found ID at rank 3 is outside the returned top-2");
  
    check(close(recall_at_k(neighbours({0, 1, 2, 3, 4, 5, 6, 7, 8, 42}),
                            {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}, 10), 0.9), "nine of ten");
    check_throws([] { (void)recall_at_k(neighbours({1}), {1}, 0); }, "k = 0");
    check_throws([] { (void)recall_at_k(neighbours({1}), {1}, 2); }, "reference shorter than k");
}

void test_evaluate_recall() {
    cvs::bench::GroundTruth truth;
    truth.k = 2;
    truth.ids = {{0, 1}, {5, 6}, {8, 9}};
    std::vector<cvs::SearchResult> results(3);
    results[0].neighbors = neighbours({0, 1});  
    results[1].neighbors = neighbours({6, 4});   
    results[2].neighbors = neighbours({1, 2});  
    const auto summary = cvs::bench::evaluate_recall(results, truth, 2);
    check(close(summary.mean, 0.5), "mean of 1, 0.5, 0");
    check(close(summary.min, 0.0), "minimum recall");
    check(summary.perfect_queries == 1, "one perfect query");
    check(summary.per_query.size() == 3 && close(summary.per_query[1], 0.5), "per-query values");
    results.pop_back();
    check_throws([&] { (void)cvs::bench::evaluate_recall(results, truth, 2); }, "query count mismatch");
}

void test_percentiles() {
    using cvs::bench::nearest_rank_percentile;
    const std::vector<double> ten{1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    check(close(nearest_rank_percentile(ten, 0.5), 5), "median of 1..10 is rank 5");
    check(close(nearest_rank_percentile(ten, 0.95), 10), "p95 of 1..10 is rank 10");
    check(close(nearest_rank_percentile(ten, 0.1), 1), "p10 is rank 1");
    check(close(nearest_rank_percentile({4.0}, 0.5), 4.0), "single sample");
    std::vector<double> hundred;
    for (int i = 1; i <= 100; ++i) hundred.push_back(i);
    check(close(nearest_rank_percentile(hundred, 0.95), 95), "p95 of 1..100");
    check(close(nearest_rank_percentile(hundred, 0.99), 99), "p99 of 1..100");
    check_throws([] { (void)nearest_rank_percentile({}, 0.5); }, "empty sample");
    check_throws([&] { (void)nearest_rank_percentile(ten, 0.0); }, "p = 0");
}

void test_ground_truth_csv() {
    const std::string header = "query_id,rank,neighbor_id,squared_distance\n";
    const auto good = write_temp("good", header +
        "0,1,4,0.5\n0,2,2,0.75\n1,1,9,0\n1,2,4,1.25\n");
    const auto truth = cvs::bench::load_ground_truth_csv(good);
    check(truth.k == 2 && truth.ids.size() == 2, "shape of a valid sheet");
    check(truth.ids[0] == std::vector<std::size_t>{4, 2}, "query 0 IDs in rank order");
    check(truth.ids[1] == std::vector<std::size_t>{9, 4}, "query 1 IDs in rank order");

    const auto crlf = write_temp("crlf", "query_id,rank,neighbor_id,squared_distance\r\n0,1,3,0.1\r\n");
    check(cvs::bench::load_ground_truth_csv(crlf).ids[0][0] == 3, "Windows line endings");

    const std::vector<std::pair<std::string, std::string>> bad = {
        {"header", "id,rank,neighbor,dist\n0,1,1,0\n"},
        {"empty", header},
        {"skip_query", header + "1,1,4,0.5\n"},
        {"skip_rank", header + "0,1,4,0.5\n0,3,2,0.7\n"},
        {"uneven", header + "0,1,4,0.5\n0,2,2,0.7\n1,1,9,0\n"},
        {"repeat", header + "0,1,4,0.5\n0,2,4,0.7\n"},
        {"negative", header + "0,1,-4,0.5\n"},
        {"columns", header + "0,1,4\n"},
        {"distance", header + "0,1,4,abc\n"},
        {"nan", header + "0,1,4,nan\n"},
    };
    for (const auto& [name, text] : bad) {
        const auto path = write_temp("bad_" + name, text);
        check_throws([&] { (void)cvs::bench::load_ground_truth_csv(path); }, "bad sheet: " + name);
        std::remove(path.c_str());
    }
    check_throws([] { (void)cvs::bench::load_ground_truth_csv("does-not-exist.csv"); }, "missing file");
    std::remove(good.c_str());
    std::remove(crlf.c_str());
}

void test_prefix_subset_keeps_ids() {
    const auto db = line_database(10);
    const auto prefix = cvs::bench::prefix_subset(db, 4);
    check(prefix.rows() == 4 && prefix.dimensions() == 2, "prefix shape");
    check(prefix.row(3)[0] == 3.0F, "row 3 is still vector 3");
    check_throws([&] { (void)cvs::bench::prefix_subset(db, 11); }, "prefix larger than database");
}

void test_latency_records() {
    const auto db = line_database(50);
    const cvs::ScalarFlatIndex index(db);
    std::vector<std::vector<float>> queries{{1.2F, 0}, {30.0F, 0}, {49.0F, 0}};
    const auto run = cvs::bench::measure_latency(index, queries, 3, 2, 4);
    check(run.records.size() == 12, "one record per query per repetition");
    check(run.records[5].repetition == 1 && run.records[5].query_id == 2, "record order");
    check(close(run.mean_candidates, 50.0), "flat search examines every vector");
    check(run.max_ms >= run.p99_ms && run.p99_ms >= run.p95_ms && run.p95_ms >= run.median_ms,
          "percentiles ordered");
    check_throws([&] { (void)cvs::bench::measure_latency(index, queries, 3, 0, 0); }, "zero repetitions");
    check_throws([&] { (void)cvs::bench::measure_latency(index, {}, 3, 0, 1); }, "no queries");
}

void test_recall_of_an_approximate_route() {
   
    const auto db = line_database(20);
    const cvs::ScalarFlatIndex exact(db);
    const FirstHalfIndex approx(db);
    const std::vector<std::vector<float>> queries{{9.4F, 0}, {2.0F, 0}};
    cvs::bench::GroundTruth truth;
    truth.k = 4;
    for (const auto& r : search_each(exact, queries, 4)) {
        truth.ids.emplace_back();
        for (const auto& n : r.neighbors) truth.ids.back().push_back(n.id);
    }
    check(truth.ids[0] == std::vector<std::size_t>{9, 10, 8, 11}, "hand-computed exact answer");
    const auto summary = cvs::bench::evaluate_recall(search_each(approx, queries, 4), truth, 4);
    check(close(summary.per_query[0], 0.5), "approximate route loses 10 and 11");
    check(close(summary.per_query[1], 1.0), "approximate route is exact far from the cut");
    check(close(summary.mean, 0.75), "mean recall 0.75");
    const auto exact_summary = cvs::bench::evaluate_recall(search_each(exact, queries, 4), truth, 4);
    check(close(exact_summary.mean, 1.0), "exact route scores 1.0 against itself");
}


void test_route_registry() {
    const auto db = line_database(16);
    const auto names = cvs::bench::available_routes();
    check(names.size() == 3, "scalar_flat, optimized_flat and ivf registered");
    for (const auto& name : names) {
        auto route = name == "ivf"
            ? cvs::bench::build_route(name, db, cvs::bench::RouteParams({{"lists", "4"}, {"probes", "4"}}))
            : cvs::bench::build_route(name, db, {});
        check(route.index && route.index->dimensions() == 2, name + " builds");
        check(route.metadata.count("backend") == 1, name + " reports its backend");
        const auto result = route.index->search({3.0F, 0.0F}, 3);
        check(result.neighbors.size() == 3 && result.neighbors[0].id == 3,
              name + " finds the right point");
    }

    const cvs::ScalarFlatIndex exact(db);
    auto all_lists = cvs::bench::build_route("ivf", db, cvs::bench::RouteParams(
        {{"lists", "4"}, {"probes", "4"}, {"backend", "scalar"}}));
    auto one_list = cvs::bench::build_route("ivf", db, cvs::bench::RouteParams(
        {{"lists", "4"}, {"probes", "1"}, {"backend", "scalar"}}));
    check(all_lists.metadata["lists"] == "4" && all_lists.metadata["probes"] == "4",
          "ivf reports lists and probes");
    for (float x : {0.2F, 7.6F, 15.0F}) {
        const auto want = exact.search({x, 0.0F}, 5);
        const auto got = all_lists.index->search({x, 0.0F}, 5);
        bool same = want.neighbors.size() == got.neighbors.size();
        for (std::size_t i = 0; same && i < want.neighbors.size(); ++i)
            same = want.neighbors[i].id == got.neighbors[i].id &&
                   want.neighbors[i].squared_distance == got.neighbors[i].squared_distance;
        check(same, "ivf probing every list matches exhaustive search");
        check(one_list.index->search({x, 0.0F}, 5).candidates_examined < 16,
              "ivf with one probe scores fewer vectors");
    }

    auto scalar_backend = cvs::bench::build_route(
        "optimized_flat", db, cvs::bench::RouteParams({{"backend", "scalar"}}));
    check(scalar_backend.metadata["backend"] == "scalar", "explicit scalar backend honoured");

    check_throws([&] { (void)cvs::bench::build_route("nope", db, {}); }, "unknown route");
    check_throws([&] {
        (void)cvs::bench::build_route("scalar_flat", db, cvs::bench::RouteParams({{"probes", "8"}}));
    }, "scalar_flat rejects stray parameters");
    check_throws([&] {
        (void)cvs::bench::build_route("optimized_flat", db, cvs::bench::RouteParams({{"backend", "gpu"}}));
    }, "bad backend name");
    check_throws([&] {
        (void)cvs::bench::build_route("ivf", db, cvs::bench::RouteParams({{"lists", "4"}, {"probes", "5"}}));
    }, "ivf probes above lists");
    check_throws([&] {
        (void)cvs::bench::build_route("ivf", db, cvs::bench::RouteParams({{"lists", "17"}}));
    }, "ivf lists above database rows");

    cvs::bench::RouteParams params;
    params.add("probes=8");
    check_throws([&] { params.add("probes=9"); }, "repeated parameter");
    check_throws([&] { params.add("novalue"); }, "parameter without '='");
    check(params.positive_integer("probes", 1) == 8, "integer parameter");
    cvs::bench::RouteParams zero({{"lists", "0"}});
    check_throws([&] { (void)zero.positive_integer("lists", 1); }, "zero is not positive");
}

void test_memory_and_machine() {
    const auto before = cvs::bench::read_memory();
#if defined(__linux__) || defined(__APPLE__) || defined(_WIN32)
    check(before.peak_rss_bytes.has_value() && *before.peak_rss_bytes > 0, "peak memory readable");
#endif
#if defined(__linux__) || defined(_WIN32)
    check(before.current_rss_bytes.has_value(), "current memory readable");
    std::vector<char> block(32u << 20, 1);
    for (std::size_t i = 0; i < block.size(); i += 4096) block[i] = static_cast<char>(i);
    const auto after = cvs::bench::read_memory();
    check(*after.peak_rss_bytes >= *before.current_rss_bytes + (24u << 20),
          "peak memory grows after touching 32 MB");
    volatile char keep = block[12345];
    (void)keep;
#endif
    (void)cvs::bench::reset_peak_memory();
    check(std::string(cvs::bench::memory_method()).size() > 10, "memory method described");
    check(!cvs::bench::cpu_model().empty(), "cpu model reported");
}

class ThrowingIndex final : public cvs::SearchIndex {
public:
    std::size_t dimensions() const noexcept override { return 1; }
    cvs::SearchResult search(const std::vector<float>& q, std::size_t) const override {
        if (q[0] == 7.0F) throw std::runtime_error("boom");
        return {};
    }
};

class CountingIndex final : public cvs::SearchIndex {
public:
    explicit CountingIndex(std::size_t n) : counts_(n) {}
    std::size_t dimensions() const noexcept override { return 1; }
    cvs::SearchResult search(const std::vector<float>& q, std::size_t) const override {
        counts_[static_cast<std::size_t>(q[0])].fetch_add(1);
        cvs::SearchResult r;
        r.neighbors.push_back({static_cast<std::size_t>(q[0]), q[0]});
        r.candidates_examined = 1;
        return r;
    }
    std::size_t count(std::size_t i) const { return counts_[i].load(); }
private:
    mutable std::vector<std::atomic<std::size_t>> counts_;
};

void test_thread_caps() {
    using cvs::bench::cap_thread_counts;
    check(cap_thread_counts({1, 2, 4}, 2) == std::vector<std::size_t>{1, 2}, "cap at 2 cores");
    check(cap_thread_counts({1, 2, 4}, 8) == std::vector<std::size_t>{1, 2, 4}, "no cap needed");
    check(cap_thread_counts({1, 2, 2, 4}, 0) == std::vector<std::size_t>{1, 2, 4}, "unknown hardware");
    check(cap_thread_counts({4, 8}, 2) == std::vector<std::size_t>{1}, "falls back to one thread");
    check_throws([] { (void)cap_thread_counts({0}, 4); }, "zero threads");
}

void test_threaded_search_matches_sequential() {
    std::vector<float> values{0.5F, 0.5F, 0.5F, 0.5F};
    for (int i = 0; i < 300; ++i) {
        values.push_back(std::sin(0.37F * i));
        values.push_back(std::cos(0.11F * i));
    }
    const cvs::DenseMatrix db(2, values);
    const cvs::ScalarFlatIndex index(db);
    std::vector<std::vector<float>> queries;
    for (int i = 0; i < 64; ++i) queries.push_back({std::cos(0.7F * i), std::sin(0.3F * i)});
    queries.push_back({0.5F, 0.5F});

    const auto one = cvs::bench::search_all(index, queries, 5, 1);
    for (std::size_t threads : {2, 3, 4, 8}) {
        const auto many = cvs::bench::search_all(index, queries, 5, threads);
        check(cvs::bench::same_results(one, many),
              "results with " + std::to_string(threads) + " threads match one thread");
    }
    for (std::size_t q = 0; q < queries.size(); ++q)
        check(cvs::bench::same_results({one[q]}, {index.search(queries[q], 5)}),
              "query " + std::to_string(q) + " result is in its own slot");
    check(one.back().neighbors[0].id == 0 && one.back().neighbors[1].id == 1,
          "duplicate vectors tie-break on smaller ID");
}

void test_each_query_answered_once() {
    const std::size_t n = 257;
    std::vector<std::vector<float>> queries;
    for (std::size_t i = 0; i < n; ++i) queries.push_back({static_cast<float>(i)});
    for (std::size_t threads : {1, 2, 4}) {
        CountingIndex index(n);
        const auto batch = cvs::bench::measure_batch(index, queries, 1, threads, 3);
        bool exact = true;
        for (std::size_t i = 0; i < n; ++i) exact = exact && index.count(i) == 3;
        check(exact, "every query answered once per repetition with " + std::to_string(threads) + " threads");
        std::size_t total = 0;
        for (auto c : batch.queries_per_thread) total += c;
        check(total == n && batch.queries_per_thread.size() == threads, "per-thread counts add up");
        check(batch.wall_ms.size() == 3 && batch.queries_per_second > 0, "batch timing recorded");
        check(batch.best_queries_per_second >= batch.queries_per_second, "best >= median throughput");
    }
}

void test_worker_exception_propagates() {
    ThrowingIndex index;
    std::vector<std::vector<float>> queries;
    for (int i = 0; i < 20; ++i) queries.push_back({static_cast<float>(i)});
    check_throws([&] { (void)cvs::bench::search_all(index, queries, 1, 4); }, "search_all rethrows");
    check_throws([&] { (void)cvs::bench::measure_batch(index, queries, 1, 4, 1); }, "batch rethrows");
}
} 

int main() {
    test_recall_by_hand();
    test_evaluate_recall();
    test_percentiles();
    test_ground_truth_csv();
    test_prefix_subset_keeps_ids();
    test_latency_records();
    test_recall_of_an_approximate_route();
    test_route_registry();
    test_memory_and_machine();
    test_thread_caps();
    test_threaded_search_matches_sequential();
    test_each_query_answered_once();
    test_worker_exception_propagates();
    if (failures != 0) {
        std::cerr << failures << " benchmark harness check(s) failed\n";
        return 1;
    }
    std::cout << "Benchmark harness checks passed.\n";
    return 0;
}
