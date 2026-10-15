#include <immintrin.h>

#include <cstddef>

namespace cvs {
namespace {

bool avx2_runtime_available_impl() noexcept {
#if defined(__GNUC__) || defined(__clang__)
    __builtin_cpu_init();
    return __builtin_cpu_supports("avx2");
#else
    return false;
#endif
}

} // namespace

bool avx2_runtime_available() noexcept {
    return avx2_runtime_available_impl();
}

float score_squared_l2_avx2(const float* a, const float* b,
                            std::size_t dimensions) noexcept {
    __m256 sums = _mm256_setzero_ps();
    std::size_t d = 0;
    for (; d + 8 <= dimensions; d += 8) {
        const __m256 av = _mm256_loadu_ps(a + d);
        const __m256 bv = _mm256_loadu_ps(b + d);
        const __m256 difference = _mm256_sub_ps(av, bv);
        sums = _mm256_add_ps(sums, _mm256_mul_ps(difference, difference));
    }

    alignas(32) float lanes[8];
    _mm256_store_ps(lanes, sums);
    float total = 0.0F;
    for (float lane : lanes) total += lane;
    for (; d < dimensions; ++d) {
        const float difference = a[d] - b[d];
        total += difference * difference;
    }
    return total;
}

} // namespace cvs
