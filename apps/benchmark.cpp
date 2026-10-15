#include "cvs/io.hpp"
#include "cvs/search.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using Clock = std::chrono::steady_clock;
double milliseconds(Clock::time_point start, Clock::time_point end) {
    return std::chrono::duration<double, std::milli>(end - start).count();
}
double percentile(const std::vector<double>& sorted, double p) {
    const auto rank = static_cast<std::size_t>(std::ceil(p * sorted.size()));
    return sorted.at(std::max<std::size_t>(1, rank) - 1);
}
}

int main(int argc, char** argv) {
    try {
        std::string database_path, query_path, output_path;
        std::size_t k = 10;
        for (int i = 1; i < argc; ++i) {
            const std::string option = argv[i];
            if (option == "--help") {
                std::cout << "Usage: vector_search --database FILE --queries FILE [--k N] [--output FILE]\n";
                return 0;
            }
            if (i + 1 >= argc) throw std::invalid_argument("missing value for " + option);
            const std::string value = argv[++i];
            if (option == "--database") database_path = value;
            else if (option == "--queries") query_path = value;
            else if (option == "--output") output_path = value;
            else if (option == "--k") {
                if (value.empty() || value.find_first_not_of("0123456789") != std::string::npos)
                    throw std::invalid_argument("k must be a positive integer");
                const auto parsed = std::stoull(value);
                k = static_cast<std::size_t>(parsed);
                if (k == 0 || k != parsed) throw std::invalid_argument("k out of range");
            } else throw std::invalid_argument("unknown option: " + option);
        }
        if (database_path.empty() || query_path.empty())
            throw std::invalid_argument("--database and --queries are required; see --help");

        const auto load_start = Clock::now();
        const auto database = cvs::load_vectors(database_path);
        const auto queries = cvs::load_vectors(query_path);
        const auto load_end = Clock::now();
        if (database.rows() == 0 || queries.rows() == 0 || k > database.rows())
            throw std::invalid_argument("need nonempty inputs and k <= database rows");
        if (database.dimensions() != queries.dimensions())
            throw std::invalid_argument("database and query dimension mismatch");
        const auto build_start = Clock::now();
        const cvs::ScalarFlatIndex index(database);
        const auto build_end = Clock::now();

        // Prepare query buffers before timing; search validation and allocations stay timed.
        std::vector<std::vector<float>> query_buffers;
        for (std::size_t q = 0; q < queries.rows(); ++q) {
            const auto* row = queries.row(q);
            query_buffers.emplace_back(row, row + queries.dimensions());
        }
        for (std::size_t q = 0; q < std::min<std::size_t>(10, query_buffers.size()); ++q)
            (void)index.search(query_buffers[q], k);

        std::vector<double> latencies;
        latencies.reserve(queries.rows());
        std::size_t candidates = 0;
        double checksum = 0;
        for (const auto& query : query_buffers) {
            const auto start = Clock::now();
            const auto result = index.search(query, k);
            const auto end = Clock::now();
            latencies.push_back(milliseconds(start, end));
            candidates += result.candidates_examined;
            for (const auto& neighbor : result.neighbors)
                checksum += neighbor.id + static_cast<double>(neighbor.squared_distance);
        }
        std::sort(latencies.begin(), latencies.end());
        // Separate sequential batch measurement excludes individual-query clock reads.
        const auto batch_start = Clock::now();
        for (const auto& query : query_buffers) {
            const auto result = index.search(query, k);
            for (const auto& neighbor : result.neighbors) checksum += neighbor.squared_distance;
        }
        const auto batch_end = Clock::now();
        const double batch_ms = milliseconds(batch_start, batch_end);
        if (batch_ms <= 0) throw std::runtime_error("timer resolution too coarse for batch");

        std::ofstream file;
        if (!output_path.empty()) {
            file.open(output_path);
            if (!file) throw std::runtime_error("cannot create output: " + output_path);
        }
        std::ostream& out = output_path.empty() ? std::cout : file;
        out << "{\n  \"schema_version\": 1,\n  \"route\": \"scalar_flat\",\n"
            << "  \"purpose\": \"smoke_benchmark\",\n  \"threads\": 1,\n"
            << "  \"database_rows\": " << database.rows() << ",\n"
            << "  \"query_rows\": " << queries.rows() << ",\n"
            << "  \"dimensions\": " << database.dimensions() << ",\n"
            << "  \"k\": " << k << ",\n"
            << "  \"load_ms\": " << milliseconds(load_start, load_end) << ",\n"
            << "  \"index_build_ms\": " << milliseconds(build_start, build_end) << ",\n"
            << "  \"latency_median_ms\": " << percentile(latencies, 0.5) << ",\n"
            << "  \"latency_p95_ms\": " << percentile(latencies, 0.95) << ",\n"
            << "  \"batch_queries_per_second\": " << 1000.0 * queries.rows() / batch_ms << ",\n"
            << "  \"candidates_per_query\": " << static_cast<double>(candidates) / queries.rows() << ",\n"
            << "  \"database_vector_bytes\": " << database.values().size() * sizeof(float) << ",\n"
            << "  \"recall_at_k\": null,\n  \"peak_memory_bytes\": null,\n"
            << "  \"checksum\": " << checksum << "\n}\n";
        out.flush();
        if (!out) throw std::runtime_error("failed to write benchmark output");
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << '\n';
        return 1;
    }
}
