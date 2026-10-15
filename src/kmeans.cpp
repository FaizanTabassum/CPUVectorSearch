#include "cvs/kmeans.hpp"

#include <numeric>
#include <random>
#include <stdexcept>
#include <vector>

namespace cvs {
namespace {

// Uniform integer in [0, bound) from raw mt19937 output.
// std::uniform_int_distribution gives different numbers on Linux (libstdc++)
// and macOS (libc++), so we do the mapping ourselves to keep seeds portable.
// Rejection sampling removes modulo bias.
std::size_t uniform_below(std::mt19937& rng, std::size_t bound) {
    const std::uint64_t range = std::uint64_t{1} << 32;  // mt19937 yields 32-bit values
    const std::uint64_t limit = range - range % bound;
    std::uint64_t value;
    do { value = rng(); } while (value >= limit);
    return static_cast<std::size_t>(value % bound);
}

// Nearest centroid; strict '<' keeps ties on the smaller index (same rule as IvfIndex).
std::size_t nearest(const float* vector, const std::vector<float>& centroids,
                    std::size_t nlist, std::size_t dims) {
    std::size_t best = 0;
    float best_distance = squared_l2(vector, centroids.data(), dims);
    for (std::size_t c = 1; c < nlist; ++c) {
        const float distance = squared_l2(vector, centroids.data() + c * dims, dims);
        if (distance < best_distance) {
            best = c;
            best_distance = distance;
        }
    }
    return best;
}

} // namespace

DenseMatrix train_kmeans(const DenseMatrix& data, std::size_t nlist,
                         std::uint32_t seed, std::size_t max_iterations) {
    const std::size_t n = data.rows();
    const std::size_t dims = data.dimensions();
    if (nlist == 0) throw std::invalid_argument("nlist must be positive");
    if (nlist > n) throw std::invalid_argument("nlist cannot exceed the number of training vectors");
    if (max_iterations == 0) throw std::invalid_argument("max_iterations must be positive");

    // 1. Initial centroids: nlist distinct rows via a seeded partial Fisher-Yates shuffle.
    std::mt19937 rng(seed);
    std::vector<std::size_t> order(n);
    std::iota(order.begin(), order.end(), std::size_t{0});
    for (std::size_t i = 0; i < nlist; ++i) {
        std::swap(order[i], order[i + uniform_below(rng, n - i)]);
    }
    std::vector<float> centroids(nlist * dims);
    for (std::size_t c = 0; c < nlist; ++c) {
        const float* row = data.row(order[c]);
        std::copy(row, row + dims, centroids.begin() + c * dims);
    }

    // Start with "unassigned" so the first round always counts as a change.
    std::vector<std::size_t> assignment(n, nlist);
    for (std::size_t iteration = 0; iteration < max_iterations; ++iteration) {
        // 2. Assign each vector to its nearest centroid.
        bool changed = false;
        for (std::size_t id = 0; id < n; ++id) {
            const std::size_t c = nearest(data.row(id), centroids, nlist, dims);
            if (c != assignment[id]) {
                assignment[id] = c;
                changed = true;
            }
        }
        if (!changed) break;  // converged

        // 3. Move each centroid to the mean of its vectors.
        //    Sum in double (fixed ID order) for accuracy, then round to float.
        std::vector<double> sums(nlist * dims, 0.0);
        std::vector<std::size_t> counts(nlist, 0);
        for (std::size_t id = 0; id < n; ++id) {
            const float* row = data.row(id);
            double* sum = sums.data() + assignment[id] * dims;
            for (std::size_t d = 0; d < dims; ++d) sum[d] += row[d];
            ++counts[assignment[id]];
        }
        for (std::size_t c = 0; c < nlist; ++c) {
            if (counts[c] == 0) continue;  // empty cluster keeps its old centroid
            for (std::size_t d = 0; d < dims; ++d) {
                centroids[c * dims + d] = static_cast<float>(sums[c * dims + d] / counts[c]);
            }
        }
    }
    return DenseMatrix(dims, std::move(centroids));
}

} // namespace cvs
