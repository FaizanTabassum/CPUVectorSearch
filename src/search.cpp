#include "cvs/search.hpp"

#include <algorithm>
#include <cmath>
#include <queue>
#include <stdexcept>
#include <utility>

namespace cvs {

DenseMatrix::DenseMatrix(std::size_t dimensions, std::vector<float> values)
    : dimensions_(dimensions), values_(std::move(values)) {
    if (dimensions_ == 0 || values_.size() % dimensions_ != 0) {
        throw std::invalid_argument("matrix dimensions must be positive and match storage");
    }
    for (float value : values_) {
        if (!std::isfinite(value)) throw std::invalid_argument("matrix contains non-finite values");
    }
}

const float* DenseMatrix::row(std::size_t id) const {
    if (id >= rows()) throw std::out_of_range("vector ID out of range");
    return values_.data() + id * dimensions_;
}

bool nearer(const Neighbor& a, const Neighbor& b) noexcept {
    return a.squared_distance < b.squared_distance ||
        (a.squared_distance == b.squared_distance && a.id < b.id);
}

float squared_l2(const float* a, const float* b, std::size_t dimensions) noexcept {
    float total = 0.0F;
    for (std::size_t d = 0; d < dimensions; ++d) {
        const float difference = a[d] - b[d];
        total += difference * difference;
    }
    return total;
}

SearchResult ScalarFlatIndex::search(const std::vector<float>& query, std::size_t k) const {
    if (query.size() != dimensions()) throw std::invalid_argument("query dimension mismatch");
    for (float value : query) {
        if (!std::isfinite(value)) throw std::invalid_argument("query contains non-finite values");
    }
    if (k == 0) return {};
    k = std::min(k, database_.rows());
    if (k == 0) return {};
    // Comparator makes the worst retained neighbor the heap top.
    std::priority_queue<Neighbor, std::vector<Neighbor>, decltype(&nearer)> heap(&nearer);
    for (std::size_t id = 0; id < database_.rows(); ++id) {
        const Neighbor candidate{id, squared_l2(database_.row(id), query.data(), dimensions())};
        if (heap.size() < k) heap.push(candidate);
        else if (nearer(candidate, heap.top())) {
            heap.pop();
            heap.push(candidate);
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
