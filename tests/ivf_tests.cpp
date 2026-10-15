#include "cvs/ivf.hpp"

#include <algorithm>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>
#include <utility>
#include <vector>

namespace {
void require(bool condition, const char* message) {
    // Not assert: these checks must also run in Release builds.
    if (!condition) throw std::runtime_error(message);
}
template <typename Function>
void rejects(Function function) {
    bool rejected = false;
    try { function(); } catch (const std::exception&) { rejected = true; }
    require(rejected, "invalid input accepted");
}

// Same IDs and bit-identical distances, in the same order.
void same_results(const cvs::SearchResult& a, const cvs::SearchResult& b, const char* message) {
    require(a.neighbors.size() == b.neighbors.size(), message);
    for (std::size_t i = 0; i < a.neighbors.size(); ++i) {
        require(a.neighbors[i].id == b.neighbors[i].id, message);
        require(a.neighbors[i].squared_distance == b.neighbors[i].squared_distance, message);
    }
}

// Expected result for a partial probe, computed without IvfIndex::search:
// rank centroids, take the first nprobe lists, brute-force their union with a full sort.
cvs::SearchResult expected_probe(const cvs::DenseMatrix& database, const cvs::IvfIndex& index,
                                 const std::vector<float>& query, std::size_t k, std::size_t nprobe,
                                 cvs::ScoringBackend backend) {
    const cvs::SquaredL2Scorer scorer(backend);
    std::vector<std::pair<float, std::size_t>> ranked;
    for (std::size_t c = 0; c < index.nlist(); ++c) {
        ranked.push_back({scorer(index.centroids().row(c), query.data(), query.size()), c});
    }
    std::sort(ranked.begin(), ranked.end());  // distance, then smaller centroid index
    cvs::SearchResult result;
    for (std::size_t p = 0; p < nprobe; ++p) {
        for (std::size_t id : index.list(ranked[p].second)) {
            result.neighbors.push_back({id, scorer(database.row(id), query.data(), query.size())});
        }
    }
    result.candidates_examined = result.neighbors.size();
    std::sort(result.neighbors.begin(), result.neighbors.end(), cvs::nearer);
    result.neighbors.resize(std::min(k, result.neighbors.size()));
    return result;
}

// Every database ID must appear in exactly one list, and lists are sorted.
void check_partition(const cvs::DenseMatrix& database, const cvs::IvfIndex& index) {
    std::vector<int> seen(database.rows(), 0);
    for (std::size_t l = 0; l < index.nlist(); ++l) {
        const auto& ids = index.list(l);
        require(std::is_sorted(ids.begin(), ids.end()), "list is not sorted by ID");
        for (std::size_t id : ids) {
            require(id < database.rows(), "list holds an invalid ID");
            ++seen[id];
        }
    }
    for (int count : seen) require(count == 1, "vector missing from lists or assigned twice");
}

void hand_checked_example() {
    // Left group near (0,0), right group near (10,0).
    const cvs::DenseMatrix database(2, {0,0, 1,0, 0,1, 10,0, 11,0, 10,1});
    const cvs::IvfIndex index(database, cvs::DenseMatrix(2, {0,0, 10,0}), 1,
                              cvs::ScoringBackend::scalar);
    require(index.list(0) == std::vector<std::size_t>({0, 1, 2}), "left list is wrong");
    require(index.list(1) == std::vector<std::size_t>({3, 4, 5}), "right list is wrong");

    const auto one = index.search({9, 0}, 2);  // probes only the right list
    require(one.candidates_examined == 3, "nprobe=1 should examine 3 vectors");
    require(one.neighbors.size() == 2 && one.neighbors[0].id == 3 && one.neighbors[1].id == 5,
            "nprobe=1 returned the wrong neighbours");

    const auto few = index.search({9, 0}, 10);  // k larger than the probed lists
    require(few.neighbors.size() == 3, "should return min(k, candidates) neighbours");
}

void ties_and_duplicates() {
    // Point 2 is equally far from both centroids: tie goes to centroid 0.
    // Points 0 and 3 are duplicates; points 1 and 4 too.
    const cvs::DenseMatrix database(1, {0, 4, 5, 0, 4, 10});
    const cvs::IvfIndex index(database, cvs::DenseMatrix(1, {0, 10}), 2,
                              cvs::ScoringBackend::scalar);
    require(index.list(0) == std::vector<std::size_t>({0, 1, 2, 3, 4}), "centroid tie not broken by smaller index");
    require(index.list(1) == std::vector<std::size_t>({5}), "right list is wrong");
    const auto result = index.search({0}, 5);
    same_results(result, cvs::ScalarFlatIndex(database).search({0}, 5), "ties/duplicates differ from flat");
    require(result.neighbors[0].id == 0 && result.neighbors[1].id == 3, "duplicate tie order wrong");

    // Query exactly between two centroids: probe tie goes to centroid 0.
    const auto probe_tie = index.search({5}, 10, 1);
    require(probe_tie.candidates_examined == 5, "probe tie not broken by smaller index");
}

void empty_lists_and_edges() {
    const cvs::DenseMatrix database(2, {0,0, 1,0, 0,1});
    // Centroid 1 is far away, so its list is empty.
    const cvs::IvfIndex index(database, cvs::DenseMatrix(2, {0,0, 100,100}), 1,
                              cvs::ScoringBackend::scalar);
    require(index.list(1).empty(), "far centroid should have an empty list");
    const auto empty = index.search({100, 100}, 3);  // probes only the empty list
    require(empty.neighbors.empty() && empty.candidates_examined == 0, "empty list not handled");
    same_results(index.search({0, 0}, 3, 2), cvs::ScalarFlatIndex(database).search({0, 0}, 3),
                 "all-lists search with an empty list differs from flat");

    const auto zero = index.search({0, 0}, 0);
    require(zero.neighbors.empty() && zero.candidates_examined == 0, "k=0 should return nothing");

    const cvs::DenseMatrix no_vectors(2, {});
    const cvs::IvfIndex empty_index(no_vectors, cvs::DenseMatrix(2, {0,0}), 1,
                                    cvs::ScoringBackend::scalar);
    const auto none = empty_index.search({0, 0}, 5);
    require(none.neighbors.empty() && none.candidates_examined == 0, "empty database not handled");
}

void invalid_inputs() {
    const cvs::DenseMatrix database(2, {0,0, 1,0});
    const cvs::DenseMatrix centroids(2, {0,0, 1,0});
    rejects([&] { cvs::IvfIndex bad(database, cvs::DenseMatrix(2, {}), 1); });       // no centroids
    rejects([&] { cvs::IvfIndex bad(database, cvs::DenseMatrix(3, {0,0,0}), 1); });  // dimension mismatch
    rejects([&] { cvs::IvfIndex bad(database, centroids, 0); });                     // nprobe = 0
    rejects([&] { cvs::IvfIndex bad(database, centroids, 3); });                     // nprobe > nlist
    const cvs::IvfIndex index(database, centroids, 1, cvs::ScoringBackend::scalar);
    rejects([&] { index.search({0, 0}, 1, 0); });
    rejects([&] { index.search({0, 0}, 1, 3); });
    rejects([&] { index.search({0}, 1); });
    rejects([&] { index.search({0, std::numeric_limits<float>::quiet_NaN()}, 0); });  // even when k = 0
    rejects([&] { index.search({0}, 0); });
    rejects([&] { index.list(2); });
}

void randomized_against_flat() {
    // Dyadic coordinates (multiples of 1/4) create many exact ties and duplicate vectors.
    std::mt19937 rng(5206);
    std::uniform_int_distribution<int> coordinate(-4, 4);
    for (std::size_t dims : {1U, 3U, 50U}) {
        std::vector<float> values(200 * dims);
        for (auto& x : values) x = coordinate(rng) / 4.0F;
        const cvs::DenseMatrix database(dims, std::move(values));
        for (std::size_t nlist : {1U, 4U, 16U}) {
            std::vector<float> c(nlist * dims);
            for (auto& x : c) x = coordinate(rng) / 4.0F;
            const cvs::DenseMatrix centroids(dims, std::move(c));

            for (const auto backend : {cvs::ScoringBackend::scalar,
                                       cvs::ScoringBackend::avx2,
                                       cvs::ScoringBackend::automatic}) {
                const cvs::IvfIndex index(database, centroids, 1, backend);
                const cvs::OptimizedFlatIndex flat(database, backend);
                check_partition(database, index);
                require(index.backend() == flat.backend(), "IVF did not resolve the requested scoring backend");

                for (int repeat = 0; repeat < 10; ++repeat) {
                    std::vector<float> query(dims);
                    for (auto& x : query) x = coordinate(rng) / 4.0F;
                    for (std::size_t k : {1U, 10U, 200U, 250U}) {
                        // Every scoring route must agree with flat search when all lists are probed.
                        const auto all = index.search(query, k, nlist);
                        same_results(all, flat.search(query, k),
                                     "all-lists IVF differs from flat search with the same backend");
                        require(all.candidates_examined == database.rows(),
                                "all-lists candidate count wrong");
                        for (std::size_t nprobe = 1; nprobe <= nlist; nprobe *= 2) {
                            const auto actual = index.search(query, k, nprobe);
                            const auto expected = expected_probe(database, index, query, k, nprobe,
                                                                 index.backend());
                            same_results(actual, expected,
                                         "partial probe differs from brute force over probed lists");
                            require(actual.candidates_examined == expected.candidates_examined,
                                    "partial probe candidate count wrong");
                        }
                    }
                }
            }
        }
    }
}
} // namespace

int main() {
    try {
        hand_checked_example();
        ties_and_duplicates();
        empty_lists_and_edges();
        invalid_inputs();
        randomized_against_flat();
        std::cout << "IVF matches exhaustive search when probing all lists; probe, tie and edge cases passed.\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
}
