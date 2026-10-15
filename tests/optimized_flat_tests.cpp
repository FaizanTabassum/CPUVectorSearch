#include "cvs/optimized_flat.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>

namespace {

void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

struct OracleNeighbor {
    std::size_t id;
    double distance;
};

std::vector<OracleNeighbor> oracle(const cvs::DenseMatrix& database,
                                   const std::vector<float>& query, std::size_t k) {
    std::vector<OracleNeighbor> result;
    for (std::size_t id = 0; id < database.rows(); ++id) {
        double distance = 0.0;
        for (std::size_t d = 0; d < database.dimensions(); ++d) {
            const double delta = static_cast<double>(database.row(id)[d]) - query[d];
            distance += delta * delta;
        }
        result.push_back({id, distance});
    }
    std::sort(result.begin(), result.end(), [](const auto& a, const auto& b) {
        return a.distance < b.distance || (a.distance == b.distance && a.id < b.id);
    });
    result.resize(std::min(k, result.size()));
    return result;
}

void compare_routes(const cvs::DenseMatrix& database, const std::vector<float>& query,
                    std::size_t k, cvs::ScoringBackend backend) {
    const cvs::OptimizedFlatIndex optimized(database, backend);
    const cvs::ScalarFlatIndex scalar(database);
    const auto actual = optimized.search(query, k);
    const auto reference = scalar.search(query, k);
    require(actual.candidates_examined == reference.candidates_examined,
            "candidate counts differ from scalar route");
    require(actual.neighbors.size() == reference.neighbors.size(), "result sizes differ");
    for (std::size_t i = 0; i < actual.neighbors.size(); ++i) {
        require(actual.neighbors[i].id == reference.neighbors[i].id,
                "neighbor IDs differ from scalar route");
        require(std::abs(actual.neighbors[i].squared_distance -
                         reference.neighbors[i].squared_distance) <= 1e-5F,
                "neighbor scores differ from scalar route");
    }
}

void compare_oracle(const cvs::DenseMatrix& database, const std::vector<float>& query,
                    std::size_t k, cvs::ScoringBackend backend) {
    const cvs::OptimizedFlatIndex optimized(database, backend);
    const auto actual = optimized.search(query, k);
    const auto expected = oracle(database, query, k);
    require(actual.neighbors.size() == expected.size(), "oracle result size differs");
    for (std::size_t i = 0; i < expected.size(); ++i) {
        require(actual.neighbors[i].id == expected[i].id, "neighbor IDs differ from FP64 oracle");
        require(std::abs(static_cast<double>(actual.neighbors[i].squared_distance) -
                         expected[i].distance) <= 2e-5 * std::max(1.0, expected[i].distance),
                "score outside FP64 tolerance");
    }
}

} // namespace

int main() {
    try {
        require(cvs::scoring_backend_available(cvs::ScoringBackend::scalar),
                "scalar scoring backend should always be available");
        require(cvs::resolve_scoring_backend(cvs::ScoringBackend::automatic) !=
                    cvs::ScoringBackend::automatic,
                "automatic backend was not resolved");
        const auto avx2_resolution =
            cvs::resolve_scoring_backend(cvs::ScoringBackend::avx2);
        require(avx2_resolution == (cvs::scoring_backend_available(cvs::ScoringBackend::avx2)
                                      ? cvs::ScoringBackend::avx2
                                      : cvs::ScoringBackend::scalar),
                "explicit AVX2 request did not resolve safely");

        // Odd dimensions exercise padding, while 50 dimensions exercise the
        // project's target shape and AVX2 tail handling.
        for (std::size_t dimensions : {1U, 3U, 7U, 8U, 9U, 50U}) {
            std::vector<float> values;
            const std::size_t rows = 19;
            values.reserve(rows * dimensions);
            for (std::size_t id = 0; id < rows; ++id) {
                for (std::size_t d = 0; d < dimensions; ++d) {
                    values.push_back(static_cast<float>(static_cast<int>((id * 7 + d * 3) % 17) - 8) /
                                     8.0F);
                }
            }
            // Duplicate rows exercise deterministic ID tie-breaking.
            std::copy_n(values.begin(), dimensions, values.begin() + 5 * dimensions);
            const cvs::DenseMatrix database(dimensions, values);
            std::vector<float> query(dimensions, 0.125F);
            for (const auto backend : {cvs::ScoringBackend::scalar,
                                       cvs::ScoringBackend::avx2,
                                       cvs::ScoringBackend::automatic}) {
                const cvs::OptimizedFlatIndex index(database, backend);
                require(index.padded_dimensions() >= dimensions &&
                            index.padded_dimensions() % 8 == 0,
                        "dimension padding is incorrect");
                for (std::size_t k : {0U, 1U, 5U, 19U, 25U}) {
                    compare_routes(database, query, k, backend);
                }
                compare_oracle(database, query, 5, backend);
            }
        }

        const cvs::DenseMatrix empty(3, {});
        const cvs::OptimizedFlatIndex empty_index(empty);
        require(empty_index.search({0, 0, 0}, 10).neighbors.empty(),
                "empty database should return no results");
        require(empty_index.search({0, 0, 0}, 10).candidates_examined == 0,
                "empty database should examine no candidates");

        const cvs::DenseMatrix one(2, {1, 0});
        const cvs::OptimizedFlatIndex one_index(one);
        bool rejected = false;
        try { (void)one_index.search({1}, 0); }
        catch (const std::invalid_argument&) { rejected = true; }
        require(rejected, "invalid query dimensions must be checked when k is zero");
        rejected = false;
        try { (void)one_index.search({1, std::numeric_limits<float>::infinity()}, 1); }
        catch (const std::invalid_argument&) { rejected = true; }
        require(rejected, "non-finite query must be rejected");

        const float a[] = {0, 1, 2, 3, 4, 5, 6, 7, 8};
        const float b[] = {1, 1, 1, 1, 1, 1, 1, 1, 1};
        const float scalar_score = cvs::score_squared_l2(a, b, 9, cvs::ScoringBackend::scalar);
        const float avx2_score = cvs::score_squared_l2(a, b, 9, cvs::ScoringBackend::avx2);
        require(std::abs(scalar_score - avx2_score) < 1e-5F,
                "explicit backend scoring differs beyond expected rounding");
        const cvs::SquaredL2Scorer reusable(cvs::ScoringBackend::avx2);
        require(reusable.backend() == avx2_resolution &&
                    std::abs(reusable(a, b, 9) - avx2_score) < 1e-5F,
                "reusable scorer did not preserve its resolved backend");

        std::cout << "Optimized flat scalar/AVX2 backends passed layout, oracle, and edge checks.\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
}
