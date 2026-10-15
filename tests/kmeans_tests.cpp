#include "cvs/ivf.hpp"
#include "cvs/kmeans.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>
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

void finds_obvious_clusters() {
    // Three well-separated blobs of 4 points; each blob's mean is exact in FP32.
    const cvs::DenseMatrix data(2, {0,0, 1,0, 0,1, 1,1,
                                    10,10, 11,10, 10,11, 11,11,
                                    20,0, 21,0, 20,1, 21,1});
    const std::vector<std::vector<float>> means = {{0.5F, 0.5F}, {10.5F, 10.5F}, {20.5F, 0.5F}};
    for (std::uint32_t seed : {5206U, 5207U, 5208U}) {
        const auto centroids = cvs::train_kmeans(data, 3, seed);
        // Every blob mean must appear exactly once among the centroids (in any order).
        for (const auto& mean : means) {
            int matches = 0;
            for (std::size_t c = 0; c < 3; ++c) {
                if (centroids.row(c)[0] == mean[0] && centroids.row(c)[1] == mean[1]) ++matches;
            }
            require(matches == 1, "k-means did not find an obvious cluster");
        }
        // The IVF lists built from these centroids are exactly the blobs.
        const cvs::IvfIndex index(data, centroids, 1);
        for (std::size_t l = 0; l < 3; ++l) {
            const auto& ids = index.list(l);
            require(ids.size() == 4 && ids[3] == ids[0] + 3 && ids[0] % 4 == 0, "lists do not match blobs");
        }
    }
}

void same_seed_same_result() {
    std::mt19937 rng(1);
    std::normal_distribution<float> noise;  // only used to build data, not compared across platforms
    std::vector<float> values(500 * 8);
    for (auto& x : values) x = noise(rng);
    const cvs::DenseMatrix data(8, std::move(values));
    for (std::uint32_t seed : {5206U, 5207U, 5208U}) {
        const auto a = cvs::train_kmeans(data, 16, seed);
        const auto b = cvs::train_kmeans(data, 16, seed);
        require(a.values() == b.values(), "same seed gave different centroids");
        const cvs::IvfIndex ia(data, a, 1), ib(data, b, 1);
        for (std::size_t l = 0; l < 16; ++l) require(ia.list(l) == ib.list(l), "same seed gave different lists");
        for (float x : a.values()) require(std::isfinite(x), "non-finite centroid");
    }
    require(cvs::train_kmeans(data, 16, 5206).values() != cvs::train_kmeans(data, 16, 5207).values(),
            "different seeds should normally give different centroids");
}

void golden_values_are_portable() {
    // Exact expected bits for seed 5206 after one round. Fails if Linux and macOS
    // (or a future code change) pick different starting rows.
    std::vector<float> values;
    for (int i = 0; i < 12; ++i) {
        values.push_back(static_cast<float>(i));
        values.push_back(static_cast<float>((i * 7) % 5));
    }
    const auto centroids = cvs::train_kmeans(cvs::DenseMatrix(2, std::move(values)), 3, 5206, 1);
    const std::vector<float> expected = {0x1.355556p+3F, 0x1p+0F, 0x1.d55556p+2F,
                                         0x1.8p+1F, 0x1.4p+1F, 0x1.aaaaaap+0F};
    require(centroids.values() == expected, "seed 5206 golden centroids changed");
}

void edge_cases() {
    // nlist == number of vectors: every vector is its own centroid.
    const cvs::DenseMatrix three(1, {0, 5, 9});
    const auto own = cvs::train_kmeans(three, 3, 5206);
    std::vector<float> sorted = own.values();
    std::sort(sorted.begin(), sorted.end());
    require(sorted == std::vector<float>({0, 5, 9}), "nlist == n should reproduce the vectors");

    // Only two distinct values but four clusters: some clusters stay empty.
    // Must not crash, and trained IVF must still match flat search when probing all lists.
    const cvs::DenseMatrix duplicates(1, {1, 1, 1, 1, 3, 3, 3, 3});
    const auto centroids = cvs::train_kmeans(duplicates, 4, 5206);
    for (float x : centroids.values()) require(std::isfinite(x), "empty cluster produced non-finite centroid");
    const cvs::IvfIndex index(duplicates, centroids, 4);
    const auto ivf = index.search({2}, 8);
    const auto flat = cvs::ScalarFlatIndex(duplicates).search({2}, 8);
    for (std::size_t i = 0; i < 8; ++i) {
        require(ivf.neighbors[i].id == flat.neighbors[i].id, "duplicate-data IVF differs from flat");
    }
}

void invalid_inputs() {
    const cvs::DenseMatrix data(2, {0,0, 1,1});
    rejects([&] { cvs::train_kmeans(data, 0, 5206); });               // no clusters
    rejects([&] { cvs::train_kmeans(data, 3, 5206); });               // more clusters than vectors
    rejects([&] { cvs::train_kmeans(data, 1, 5206, 0); });            // zero iterations
    rejects([&] { cvs::train_kmeans(cvs::DenseMatrix(2, {}), 1, 5206); });  // empty data
}

void trained_ivf_matches_flat() {
    // End to end: train on the database, then all-lists IVF == exhaustive search.
    std::mt19937 rng(7);
    std::uniform_int_distribution<int> coordinate(-8, 8);
    std::vector<float> values(300 * 5);
    for (auto& x : values) x = coordinate(rng) / 8.0F;
    const cvs::DenseMatrix data(5, std::move(values));
    const cvs::IvfIndex index(data, cvs::train_kmeans(data, 8, 5206), 8);
    const cvs::ScalarFlatIndex flat(data);
    for (int q = 0; q < 20; ++q) {
        std::vector<float> query(5);
        for (auto& x : query) x = coordinate(rng) / 8.0F;
        const auto a = index.search(query, 10);
        const auto b = flat.search(query, 10);
        require(a.candidates_examined == data.rows(), "all-lists candidate count wrong");
        for (std::size_t i = 0; i < 10; ++i) {
            require(a.neighbors[i].id == b.neighbors[i].id &&
                    a.neighbors[i].squared_distance == b.neighbors[i].squared_distance,
                    "trained all-lists IVF differs from flat search");
        }
    }
}
} // namespace

int main() {
    try {
        finds_obvious_clusters();
        same_seed_same_result();
        golden_values_are_portable();
        edge_cases();
        invalid_inputs();
        trained_ivf_matches_flat();
        std::cout << "k-means is seeded and reproducible; trained IVF matches exhaustive search.\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
}
