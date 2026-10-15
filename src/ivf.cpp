#include "cvs/ivf.hpp"

#include <stdexcept>
#include <utility>

namespace cvs {

IvfIndex::IvfIndex(const DenseMatrix& database, DenseMatrix centroids, std::size_t default_nprobe)
    : database_(database), centroids_(std::move(centroids)), default_nprobe_(default_nprobe) {
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
        float best_distance = squared_l2(vector, centroids_.row(0), dims);
        for (std::size_t c = 1; c < nlist(); ++c) {
            const float distance = squared_l2(vector, centroids_.row(c), dims);
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

// search() is the next step.

} // namespace cvs
