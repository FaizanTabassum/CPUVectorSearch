#pragma once

#include "cvs/search.hpp"

#include <cstddef>
#include <vector>

namespace cvs {

// The backend used to score a candidate vector. `automatic` resolves to AVX2
// when available and otherwise to the scalar implementation.
enum class ScoringBackend {
    automatic,
    scalar,
    avx2,
};

// Returns whether a backend can execute on this machine. `automatic` is always
// available; an unavailable explicit AVX2 request safely falls back to scalar
// when scoring.
bool scoring_backend_available(ScoringBackend backend) noexcept;
ScoringBackend resolve_scoring_backend(ScoringBackend requested) noexcept;

// Resolve a backend once and reuse the scorer across candidate IDs. This keeps
// CPU feature detection out of the per-candidate loop in flat and IVF routes.
class SquaredL2Scorer {
public:
    explicit SquaredL2Scorer(ScoringBackend backend = ScoringBackend::automatic) noexcept;

    float operator()(const float* a, const float* b, std::size_t dimensions) const noexcept;
    ScoringBackend backend() const noexcept { return backend_; }

private:
    ScoringBackend backend_;
};

// Reusable by other routes, including IVF, so both routes can use the same
// score arithmetic for exhaustive-equivalence checks. Prefer a persistent
// SquaredL2Scorer when scoring many candidates.
float score_squared_l2(const float* a, const float* b, std::size_t dimensions,
                       ScoringBackend backend = ScoringBackend::automatic) noexcept;

// Exhaustive search over a private, tiled and dimension-padded copy of the
// database. The database must outlive this index and remain immutable.
class OptimizedFlatIndex final : public SearchIndex {
public:
    explicit OptimizedFlatIndex(const DenseMatrix& database,
                                ScoringBackend backend = ScoringBackend::automatic);
    OptimizedFlatIndex(DenseMatrix&&, ScoringBackend = ScoringBackend::automatic) = delete;

    std::size_t dimensions() const noexcept override { return dimensions_; }
    SearchResult search(const std::vector<float>& query, std::size_t k) const override;

    ScoringBackend backend() const noexcept { return scorer_.backend(); }
    std::size_t padded_dimensions() const noexcept { return padded_dimensions_; }

private:
    const DenseMatrix& database_;
    std::size_t dimensions_;
    std::size_t padded_dimensions_;
    SquaredL2Scorer scorer_;
    std::vector<float> blocked_values_;
};

} // namespace cvs
