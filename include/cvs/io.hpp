#pragma once

#include "cvs/search.hpp"
#include <string>

namespace cvs {
// CVEC v1: 8-byte magic, little-endian u32 rows/dimensions, row-major FP32.
DenseMatrix load_vectors(const std::string& path);
} // namespace cvs
