#include "cvs/io.hpp"

#include <array>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <utility>

namespace cvs {
namespace {
std::uint32_t read_u32(std::istream& stream) {
    std::array<unsigned char, 4> bytes{};
    if (!stream.read(reinterpret_cast<char*>(bytes.data()), bytes.size())) {
        throw std::runtime_error("truncated CVEC file");
    }
    return std::uint32_t(bytes[0]) | (std::uint32_t(bytes[1]) << 8) |
           (std::uint32_t(bytes[2]) << 16) | (std::uint32_t(bytes[3]) << 24);
}
} // namespace

DenseMatrix load_vectors(const std::string& path) {
    static_assert(sizeof(float) == 4 && std::numeric_limits<float>::is_iec559,
                  "CVEC requires IEEE-754 binary32 floats");
    std::ifstream stream(path, std::ios::binary | std::ios::ate);
    if (!stream) throw std::runtime_error("cannot open vector file: " + path);
    const std::streamoff length = stream.tellg();
    stream.seekg(0);
    std::array<char, 8> magic{};
    if (!stream.read(magic.data(), magic.size()) ||
        magic != std::array<char, 8>{'C','V','E','C','0','0','0','1'}) {
        throw std::runtime_error("invalid CVEC magic/version: " + path);
    }
    const auto rows = read_u32(stream);
    const auto dimensions = read_u32(stream);
    if (dimensions == 0) throw std::runtime_error("CVEC dimensions must be positive");
    const std::uint64_t count = std::uint64_t(rows) * dimensions;
    // Validate using division before multiplying by four (untrusted u32 header).
    if (length < 16 || (length - 16) % 4 != 0 ||
        count != static_cast<std::uint64_t>((length - 16) / 4) ||
        count > std::vector<float>().max_size()) {
        throw std::runtime_error("CVEC payload size does not match header: " + path);
    }
    std::vector<float> values(static_cast<std::size_t>(count));
    for (float& value : values) {
        const std::uint32_t bits = read_u32(stream);
        std::memcpy(&value, &bits, sizeof(value));
    }
    return DenseMatrix(dimensions, std::move(values));
}
} // namespace cvs
