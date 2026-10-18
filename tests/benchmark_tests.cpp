

#include "cvs/benchmark.hpp"

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

} 

int main() {
    test_recall_by_hand();
    test_evaluate_recall();
    test_percentiles();
    test_ground_truth_csv();
    test_prefix_subset_keeps_ids();
    test_latency_records();
    test_recall_of_an_approximate_route();
    if (failures != 0) {
        std::cerr << failures << " benchmark harness check(s) failed\n";
        return 1;
    }
    std::cout << "Benchmark harness checks passed.\n";
    return 0;
}
