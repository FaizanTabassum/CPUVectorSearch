#pragma once

#include <cstddef>
#include <vector>

namespace cvs {

// Row-major owned FP32 storage. IDs are zero-based row positions.
class DenseMatrix {
public:
    DenseMatrix(std::size_t dimensions, std::vector<float> values);
    std::size_t rows() const noexcept { return values_.size() / dimensions_; }
    std::size_t dimensions() const noexcept { return dimensions_; }
    const float* row(std::size_t id) const;
    const std::vector<float>& values() const noexcept { return values_; }
private:
    std::size_t dimensions_;
    std::vector<float> values_;
};

struct Neighbor {
    std::size_t id;
    float squared_distance;
};

// Strict ascending order: distance first, then smaller ID.
bool nearer(const Neighbor& a, const Neighbor& b) noexcept;
float squared_l2(const float* a, const float* b, std::size_t dimensions) noexcept;

struct SearchResult {
    std::vector<Neighbor> neighbors;
    std::size_t candidates_examined = 0;
};

// Future optimised-flat and IVF routes implement this same query contract.
class SearchIndex {
public:
    virtual ~SearchIndex() = default;
    virtual std::size_t dimensions() const noexcept = 0;
    virtual SearchResult search(const std::vector<float>& query, std::size_t k) const = 0;
};

// Borrows the database: it must outlive the index and remain immutable.
class ScalarFlatIndex final : public SearchIndex {
public:
    explicit ScalarFlatIndex(const DenseMatrix& database) : database_(database) {}
    ScalarFlatIndex(DenseMatrix&&) = delete;
    std::size_t dimensions() const noexcept override { return database_.dimensions(); }
    SearchResult search(const std::vector<float>& query, std::size_t k) const override;
private:
    const DenseMatrix& database_;
};

} // namespace cvs
