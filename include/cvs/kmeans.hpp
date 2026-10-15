#pragma once

#include "cvs/search.hpp"

#include <cstddef>
#include <cstdint>

namespace cvs {

// Seeded k-means (Lloyd's algorithm) for IVF centroids.
//
// 1. Start from `nlist` distinct database rows chosen by `seed`.
// 2. Assign each vector to its nearest centroid (ties -> smaller index).
// 3. Move each centroid to the mean of its vectors; an empty cluster keeps
//    its previous centroid.
// 4. Repeat until no assignment changes or `max_iterations` is reached.
//
// Train on database vectors only. The same data, nlist, seed and
// max_iterations always give bit-identical centroids on every platform.
DenseMatrix train_kmeans(const DenseMatrix& data, std::size_t nlist,
                         std::uint32_t seed, std::size_t max_iterations = 25);

} // namespace cvs
