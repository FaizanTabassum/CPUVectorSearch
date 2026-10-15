#include "cvs/optimized_flat.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <queue>
#include <stdexcept>
#include <utility>

namespace cvs {
#if defined(CVS_HAS_AVX2_IMPL)
bool avx2_runtime_available() noexcept;
float score_squared_l2_avx2(const float*, const float*, std::size_t) noexcept;
#endif

namespace {

constexpr std::size_t kBlockRows = 8;
constexpr std::size_t kDimensionAlignment = 8;

float score_squared_l2_scalar(const float* a, const float* b,
                              std::size_t dimensions) noexcept {
    float total = 0.0F;
    for (std::size_t d = 0; d < dimensions; ++d) {
        const float difference = a[d] - b[d];
        total += difference * difference;
    }
    return total;
}

void validate_query(const std::vector<float>& query, std::size_t dimensions) {
    if (query.size() != dimensions) throw std::invalid_argument("query dimension mismatch");
    for (float value : query) {
        if (!std::isfinite(value)) throw std::invalid_argument("query contains non-finite values");
    }
}

} // namespace

bool scoring_backend_available(ScoringBackend backend) noexcept {
    switch (backend) {
    case ScoringBackend::automatic:
    case ScoringBackend::scalar:
        return true;
    case ScoringBackend::avx2:
#if defined(CVS_HAS_AVX2_IMPL)
        return avx2_runtime_available();
#else
        return false;
#endif
    }
    return false;
}

ScoringBackend resolve_scoring_backend(ScoringBackend requested) noexcept {
    if (requested == ScoringBackend::automatic) {
        return scoring_backend_available(ScoringBackend::avx2)
            ? ScoringBackend::avx2 : ScoringBackend::scalar;
    }
    if (requested == ScoringBackend::avx2 &&
        !scoring_backend_available(ScoringBackend::avx2)) {
        return ScoringBackend::scalar;
    }
    return requested;
}

SquaredL2Scorer::SquaredL2Scorer(ScoringBackend backend) noexcept
    : backend_(resolve_scoring_backend(backend)) {}

float SquaredL2Scorer::operator()(const float* a, const float* b,
                                 std::size_t dimensions) const noexcept {
#if defined(CVS_HAS_AVX2_IMPL)
    if (backend_ == ScoringBackend::avx2) {
        return score_squared_l2_avx2(a, b, dimensions);
    }
#endif
    return score_squared_l2_scalar(a, b, dimensions);
}

float score_squared_l2(const float* a, const float* b, std::size_t dimensions,
                       ScoringBackend backend) noexcept {
    return SquaredL2Scorer(backend)(a, b, dimensions);
}

OptimizedFlatIndex::OptimizedFlatIndex(const DenseMatrix& database, ScoringBackend backend)
    : database_(database), dimensions_(database.dimensions()),
      padded_dimensions_(0), scorer_(backend) {
    if (dimensions_ > std::numeric_limits<std::size_t>::max() - (kDimensionAlignment - 1)) {
        throw std::length_error("database dimensions are too large to pad");
    }
    padded_dimensions_ = ((dimensions_ + kDimensionAlignment - 1) / kDimensionAlignment) *
                         kDimensionAlignment;
    if (padded_dimensions_ != 0 &&
        database_.rows() > blocked_values_.max_size() / padded_dimensions_) {
        throw std::length_error("blocked database is too large");
    }
    blocked_values_.assign(database_.rows() * padded_dimensions_, 0.0F);

    // Each tile contains eight consecutive database IDs. Rows retain their ID
    // order, and zero-filled padding makes each row's stride SIMD-friendly.
    for (std::size_t id = 0; id < database_.rows(); ++id) {
        auto* destination = blocked_values_.data() + id * padded_dimensions_;
        std::copy_n(database_.row(id), dimensions_, destination);
    }
}

SearchResult OptimizedFlatIndex::search(const std::vector<float>& query, std::size_t k) const {
    validate_query(query, dimensions_);
    if (k == 0 || database_.rows() == 0) return {};
    k = std::min(k, database_.rows());

    std::priority_queue<Neighbor, std::vector<Neighbor>, decltype(&nearer)> heap(&nearer);
    for (std::size_t block = 0; block < database_.rows(); block += kBlockRows) {
        const std::size_t block_end = std::min(block + kBlockRows, database_.rows());
        for (std::size_t id = block; id < block_end; ++id) {
            const float* candidate = blocked_values_.data() + id * padded_dimensions_;
            const Neighbor neighbor{id, scorer_(candidate, query.data(), dimensions_)};
            if (heap.size() < k) heap.push(neighbor);
            else if (nearer(neighbor, heap.top())) {
                heap.pop();
                heap.push(neighbor);
            }
        }
    }

    SearchResult result;
    result.candidates_examined = database_.rows();
    result.neighbors.reserve(k);
    while (!heap.empty()) {
        result.neighbors.push_back(heap.top());
        heap.pop();
    }
    std::sort(result.neighbors.begin(), result.neighbors.end(), nearer);
    return result;
}

} // namespace cvs
