#pragma once

#include "cvs/search.hpp"

#include <cstddef>
#include <vector>

namespace cvs {

// Inverted-file (IVF) index.
//
// Build: each database vector goes into the list of its nearest centroid
//        (ties -> smaller centroid index). Lists store original database IDs.
// Query: score the query against every centroid, probe the `nprobe` nearest
//        lists (ties -> smaller centroid index), scan only their vectors, and
//        keep the top-k with the shared `nearer` order.
//
// With nprobe == nlist() this returns exactly the same IDs and distances as
// ScalarFlatIndex, because both use cvs::squared_l2 and cvs::nearer.
class IvfIndex final : public SearchIndex {
public:
    // Borrows the database (must outlive the index and stay immutable).
    // Centroids are copied. `default_nprobe` is used by search(query, k).
    IvfIndex(const DenseMatrix& database, DenseMatrix centroids, std::size_t default_nprobe);
    IvfIndex(DenseMatrix&&, DenseMatrix, std::size_t) = delete;

    std::size_t dimensions() const noexcept override { return database_.dimensions(); }
    SearchResult search(const std::vector<float>& query, std::size_t k) const override;
    SearchResult search(const std::vector<float>& query, std::size_t k, std::size_t nprobe) const;

    std::size_t nlist() const noexcept { return centroids_.rows(); }
    std::size_t default_nprobe() const noexcept { return default_nprobe_; }
    const DenseMatrix& centroids() const noexcept { return centroids_; }
    // Database IDs in list `list_id`, ascending.
    const std::vector<std::size_t>& list(std::size_t list_id) const;

private:
    void check_nprobe(std::size_t nprobe) const;

    const DenseMatrix& database_;
    DenseMatrix centroids_;
    std::size_t default_nprobe_;
    std::vector<std::vector<std::size_t>> lists_;
};

} // namespace cvs
