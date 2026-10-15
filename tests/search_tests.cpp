#include "cvs/search.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>
#include <utility>
#include <vector>

namespace {
void require(bool condition, const char* message) {
    // Do not use assert: correctness checks must also run in Release builds.
    if (!condition) throw std::runtime_error(message);
}
template <typename Function>
void rejects(Function function) {
    bool rejected = false;
    try { function(); } catch (const std::exception&) { rejected = true; }
    require(rejected, "invalid input accepted");
}
struct ReferenceNeighbor { std::size_t id; double distance; };

// Independent FP64 arithmetic and full sort; no production distance/comparator/top-k helpers.
std::vector<ReferenceNeighbor> oracle(const cvs::DenseMatrix& database,
                                     const std::vector<float>& query, std::size_t k) {
    std::vector<ReferenceNeighbor> all;
    for (std::size_t id = 0; id < database.rows(); ++id) {
        double sum = 0;
        for (std::size_t d = 0; d < database.dimensions(); ++d) {
            const double delta = static_cast<double>(database.row(id)[d]) - static_cast<double>(query[d]);
            sum += delta * delta;
        }
        all.push_back({id, sum});
    }
    std::sort(all.begin(), all.end(), [](const auto& a, const auto& b) {
        return a.distance < b.distance || (a.distance == b.distance && a.id < b.id);
    });
    all.resize(std::min(k, all.size()));
    return all;
}

void compare(const cvs::DenseMatrix& database, const std::vector<float>& query, std::size_t k) {
    const cvs::ScalarFlatIndex index(database);
    const auto actual = index.search(query, k);
    const auto expected = oracle(database, query, k);
    require(actual.neighbors.size() == expected.size(), "wrong result count");
    require(actual.candidates_examined == (k == 0 ? 0 : database.rows()), "wrong candidate count");
    for (std::size_t i = 0; i < expected.size(); ++i) {
        require(actual.neighbors[i].id == expected[i].id, "neighbor differs from FP64 oracle");
        require(std::abs(actual.neighbors[i].squared_distance - expected[i].distance) <=
                2e-5 * std::max(1.0, expected[i].distance), "FP32 distance outside tolerance");
    }
}
}

int main() {
    try {
        const cvs::DenseMatrix unit(2, {1,0, 0,1, -1,0, 0,-1, 1,0});
        for (std::size_t k : {0U, 1U, 3U, 5U, 9U}) compare(unit, {1,0}, k);
        const auto ties = cvs::ScalarFlatIndex(unit).search({1,0}, 5);
        require(ties.neighbors[0].id == 0 && ties.neighbors[1].id == 4 &&
                ties.neighbors[2].id == 1 && ties.neighbors[3].id == 3, "tie order is unstable");
        compare(cvs::DenseMatrix(2, {}), {1,0}, 10);
        rejects([] { cvs::DenseMatrix bad(0, {}); });
        rejects([] { cvs::DenseMatrix bad(2, {1}); });
        rejects([] { cvs::DenseMatrix bad(1, {std::numeric_limits<float>::infinity()}); });
        rejects([&] { unit.row(5); });
        const cvs::ScalarFlatIndex index(unit);
        rejects([&] { index.search({1}, 10); });
        rejects([&] { index.search({1, std::numeric_limits<float>::quiet_NaN()}, 10); });

        // Dyadic coordinates keep FP32/FP64 rankings exactly comparable, including ties.
        // This avoids asserting identical rankings for arbitrary near-ties rounded in FP32.
        std::mt19937 rng(5206);
        std::uniform_int_distribution<int> coordinate(-16, 16);
        for (std::size_t dimensions : {1U, 3U, 50U}) {
            std::vector<float> values(67 * dimensions);
            for (auto& x : values) x = coordinate(rng) / 16.0F;
            const cvs::DenseMatrix database(dimensions, std::move(values));
            for (int repeat = 0; repeat < 12; ++repeat) {
                std::vector<float> query(dimensions);
                for (auto& x : query) x = coordinate(rng) / 16.0F;
                for (std::size_t k : {1U, 10U, 67U, 80U}) compare(database, query, k);
            }
        }
        // Real-valued unit vectors exercise normalized production inputs.
        compare(cvs::DenseMatrix(2, {0.6F,0.8F, -0.6F,0.8F, 0,-1}), {1,0}, 3);
        std::cout << "Scalar search agrees with independent FP64 oracle; edge cases passed.\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
}
