#include "cvs/ivf.hpp"

#include <algorithm>
#include <cmath>
#include <queue>
#include <stdexcept>
#include <utility>

namespace cvs {

IvfIndex::IvfIndex(const DenseMatrix& database, DenseMatrix centroids,
                   std::size_t default_nprobe, ScoringBackend backend)
    : database_(database), centroids_(std::move(centroids)),
      default_nprobe_(default_nprobe), scorer_(backend) {
    // 1. Validate inputs.
    if (centroids_.rows() == 0) throw std::invalid_argument("IVF needs at least one centroid");
    if (centroids_.dimensions() != database_.dimensions()) {
        throw std::invalid_argument("centroid and database dimensions differ");
    }
    check_nprobe(default_nprobe_);

    // 2. Assign every database vector to its nearest centroid.
    //    Strict '<' means ties go to the smaller centroid index (deterministic).
    //    IDs are pushed in increasing order, so each list stays sorted.
    lists_.resize(nlist());
    const std::size_t dims = dimensions();
    for (std::size_t id = 0; id < database_.rows(); ++id) {
        const float* vector = database_.row(id);
        std::size_t best = 0;
        float best_distance = scorer_(vector, centroids_.row(0), dims);
        for (std::size_t c = 1; c < nlist(); ++c) {
            const float distance = scorer_(vector, centroids_.row(c), dims);
            if (distance < best_distance) {
                best = c;
                best_distance = distance;
            }
        }
        lists_[best].push_back(id);  // store the original database ID
    }
}

const std::vector<std::size_t>& IvfIndex::list(std::size_t list_id) const {
    if (list_id >= nlist()) throw std::out_of_range("IVF list ID out of range");
    return lists_[list_id];
}

void IvfIndex::check_nprobe(std::size_t nprobe) const {
    if (nprobe == 0 || nprobe > nlist()) {
        throw std::invalid_argument("nprobe must be between 1 and the number of lists");
    }
}

SearchResult IvfIndex::search(const std::vector<float>& query, std::size_t k) const {
    return search(query, k, default_nprobe_);
}

SearchResult IvfIndex::search(const std::vector<float>& query, std::size_t k, std::size_t nprobe) const {
    // 1. Validate, even when k == 0 (same rule as the scalar index).
    if (query.size() != dimensions()) throw std::invalid_argument("query dimension mismatch");
    for (float value : query) {
        if (!std::isfinite(value)) throw std::invalid_argument("query contains non-finite values");
    }
    check_nprobe(nprobe);
    if (k == 0) return {};

    // 2. Pick the nprobe nearest centroids. Neighbor.id holds the centroid
    //    index here, so `nearer` breaks ties toward the smaller index.
    std::vector<Neighbor> centroid_scores(nlist());
    for (std::size_t c = 0; c < nlist(); ++c) {
        centroid_scores[c] = {c, scorer_(centroids_.row(c), query.data(), dimensions())};
    }
    std::partial_sort(centroid_scores.begin(), centroid_scores.begin() + nprobe,
                      centroid_scores.end(), nearer);

    // 3. Scan only the probed lists, keeping the best k in a bounded heap.
    //    Reuse the selected backend for exact equivalence with flat search.
    std::priority_queue<Neighbor, std::vector<Neighbor>, decltype(&nearer)> heap(&nearer);
    SearchResult result;
    for (std::size_t p = 0; p < nprobe; ++p) {
        for (std::size_t id : lists_[centroid_scores[p].id]) {
            const Neighbor candidate{id, scorer_(database_.row(id), query.data(), dimensions())};
            ++result.candidates_examined;  // database vectors only, not centroids
            if (heap.size() < k) heap.push(candidate);
            else if (nearer(candidate, heap.top())) {
                heap.pop();
                heap.push(candidate);
            }
        }
    }

    // 4. Empty the heap and sort best-first.
    result.neighbors.reserve(heap.size());
    while (!heap.empty()) {
        result.neighbors.push_back(heap.top());
        heap.pop();
    }
    std::sort(result.neighbors.begin(), result.neighbors.end(), nearer);
    return result;
}

} // namespace cvs
