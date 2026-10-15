#include "cvs/io.hpp"
#include "cvs/optimized_flat.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using Clock = std::chrono::steady_clock;

struct Options {
    std::string database_path;
    std::string query_path;
    std::string output_path;
    std::string machine_model;
    std::string data_seed = "unspecified";
    std::size_t k = 10;
    std::size_t warmup_queries = 5;
    std::size_t repetitions = 5;
};

struct TimingSummary {
    double median_ms = 0.0;
    double p95_ms = 0.0;
    double queries_per_second = 0.0;
};

struct RouteSummary {
    std::string name;
    cvs::ScoringBackend backend = cvs::ScoringBackend::scalar;
    double build_ms = 0.0;
    std::size_t index_storage_bytes = 0;
    TimingSummary timing;
    std::size_t queries_with_id_differences = 0;
    std::size_t different_neighbor_slots = 0;
    double max_ranked_distance_delta = 0.0;
};

double milliseconds(Clock::time_point start, Clock::time_point end) {
    return std::chrono::duration<double, std::milli>(end - start).count();
}

std::size_t parse_size(const std::string& value, const char* name, bool allow_zero = false) {
    if (value.empty() || value.find_first_not_of("0123456789") != std::string::npos) {
        throw std::invalid_argument(std::string(name) + " must be an integer");
    }
    const auto parsed = std::stoull(value);
    const auto converted = static_cast<std::size_t>(parsed);
    if (converted != parsed || (!allow_zero && converted == 0)) {
        throw std::invalid_argument(std::string(name) + " is out of range");
    }
    return converted;
}

Options parse_options(int argc, char** argv) {
    Options options;
    for (int i = 1; i < argc; ++i) {
        const std::string option = argv[i];
        if (option == "--help") {
            std::cout << "Usage: flat_ablation --database FILE --queries FILE [options]\n"
                         "  --k N                  top-k value (default 10)\n"
                         "  --warmup-queries N     warmup queries per route (default 5)\n"
                         "  --repetitions N        full query passes per route (default 5)\n"
                         "  --data-seed VALUE      dataset seed recorded in JSON metadata\n"
                         "  --machine-model LABEL  optional CPU model for result metadata\n"
                         "  --output FILE          write JSON to FILE instead of stdout\n";
            std::exit(0);
        }
        if (i + 1 >= argc) throw std::invalid_argument("missing value for " + option);
        const std::string value = argv[++i];
        if (option == "--database") options.database_path = value;
        else if (option == "--queries") options.query_path = value;
        else if (option == "--output") options.output_path = value;
        else if (option == "--machine-model") options.machine_model = value;
        else if (option == "--data-seed") options.data_seed = value;
        else if (option == "--k") options.k = parse_size(value, "k");
        else if (option == "--warmup-queries")
            options.warmup_queries = parse_size(value, "warmup query count", true);
        else if (option == "--repetitions")
            options.repetitions = parse_size(value, "repetition count");
        else throw std::invalid_argument("unknown option: " + option);
    }
    if (options.database_path.empty() || options.query_path.empty()) {
        throw std::invalid_argument("--database and --queries are required; see --help");
    }
    return options;
}

const char* backend_name(cvs::ScoringBackend backend) {
    return backend == cvs::ScoringBackend::avx2 ? "avx2" : "scalar";
}

const char* operating_system() {
#if defined(_WIN32)
    return "windows";
#elif defined(__APPLE__)
    return "macos";
#elif defined(__linux__)
    return "linux";
#else
    return "unknown";
#endif
}

const char* architecture() {
#if defined(__x86_64__) || defined(_M_X64)
    return "x86_64";
#elif defined(__i386__) || defined(_M_IX86)
    return "x86";
#elif defined(__aarch64__) || defined(_M_ARM64)
    return "arm64";
#elif defined(__arm__) || defined(_M_ARM)
    return "arm";
#else
    return "unknown";
#endif
}

const char* build_mode() {
#if defined(NDEBUG)
    return "release";
#else
    return "debug";
#endif
}

std::string json_escape(const std::string& value) {
    std::string escaped;
    escaped.reserve(value.size());
    for (char c : value) {
        if (c == '\\' || c == '"') escaped.push_back('\\');
        if (c == '\n') escaped += "\\n";
        else if (c == '\r') escaped += "\\r";
        else escaped.push_back(c);
    }
    return escaped;
}

double percentile(const std::vector<double>& sorted, double p) {
    const auto rank = static_cast<std::size_t>(std::ceil(p * sorted.size()));
    return sorted.at(std::max<std::size_t>(1, rank) - 1);
}

template <typename SearchFunction>
TimingSummary measure(SearchFunction search,
                      const std::vector<std::vector<float>>& queries,
                      std::size_t warmup_queries, std::size_t repetitions) {
    const auto warmups = std::min(warmup_queries, queries.size());
    volatile double checksum = 0.0;
    for (std::size_t i = 0; i < warmups; ++i) {
        const auto result = search(queries[i]);
        if (!result.neighbors.empty()) checksum += result.neighbors.front().squared_distance;
    }

    std::vector<double> latencies;
    latencies.reserve(queries.size() * repetitions);
    for (std::size_t repetition = 0; repetition < repetitions; ++repetition) {
        for (const auto& query : queries) {
            const auto start = Clock::now();
            const auto result = search(query);
            const auto end = Clock::now();
            latencies.push_back(milliseconds(start, end));
            if (!result.neighbors.empty()) checksum += result.neighbors.front().squared_distance;
        }
    }
    const auto batch_start = Clock::now();
    for (std::size_t repetition = 0; repetition < repetitions; ++repetition) {
        for (const auto& query : queries) {
            const auto result = search(query);
            if (!result.neighbors.empty()) checksum += result.neighbors.front().squared_distance;
        }
    }
    const auto batch_end = Clock::now();
    if (checksum < 0.0) std::cerr << checksum;
    std::sort(latencies.begin(), latencies.end());
    const double batch_ms = milliseconds(batch_start, batch_end);
    return {percentile(latencies, 0.5), percentile(latencies, 0.95),
            batch_ms > 0.0 ? 1000.0 * latencies.size() / batch_ms : 0.0};
}

std::vector<std::vector<float>> copy_queries(const cvs::DenseMatrix& matrix) {
    std::vector<std::vector<float>> queries;
    queries.reserve(matrix.rows());
    for (std::size_t id = 0; id < matrix.rows(); ++id) {
        const auto* row = matrix.row(id);
        queries.emplace_back(row, row + matrix.dimensions());
    }
    return queries;
}

std::vector<std::vector<cvs::Neighbor>> collect_results(
    const cvs::SearchIndex& index, const std::vector<std::vector<float>>& queries,
    std::size_t k) {
    std::vector<std::vector<cvs::Neighbor>> results;
    results.reserve(queries.size());
    for (const auto& query : queries) results.push_back(index.search(query, k).neighbors);
    return results;
}

void compare_to_reference(RouteSummary& route,
                          const std::vector<std::vector<cvs::Neighbor>>& actual,
                          const std::vector<std::vector<cvs::Neighbor>>& reference) {
    for (std::size_t q = 0; q < actual.size(); ++q) {
        bool query_differs = false;
        const auto count = std::min(actual[q].size(), reference[q].size());
        for (std::size_t rank = 0; rank < count; ++rank) {
            if (actual[q][rank].id != reference[q][rank].id) {
                ++route.different_neighbor_slots;
                query_differs = true;
            }
            route.max_ranked_distance_delta = std::max(
                route.max_ranked_distance_delta,
                std::abs(static_cast<double>(actual[q][rank].squared_distance) -
                         reference[q][rank].squared_distance));
        }
        if (actual[q].size() != reference[q].size()) query_differs = true;
        if (query_differs) ++route.queries_with_id_differences;
    }
}

RouteSummary measure_scalar(const cvs::ScalarFlatIndex& index,
                            const std::vector<std::vector<float>>& queries,
                            std::size_t k, std::size_t warmups, std::size_t repetitions) {
    RouteSummary route;
    route.name = "scalar_flat";
    route.backend = cvs::ScoringBackend::scalar;
    route.timing = measure([&](const auto& query) { return index.search(query, k); },
                           queries, warmups, repetitions);
    return route;
}

RouteSummary measure_optimized(const cvs::DenseMatrix& database,
                               const std::vector<std::vector<float>>& queries,
                               std::size_t k, std::size_t warmups, std::size_t repetitions,
                               cvs::ScoringBackend requested, const char* name) {
    const auto build_start = Clock::now();
    const cvs::OptimizedFlatIndex index(database, requested);
    const auto build_end = Clock::now();
    RouteSummary route;
    route.name = name;
    route.backend = index.backend();
    route.build_ms = milliseconds(build_start, build_end);
    route.index_storage_bytes = database.rows() * index.padded_dimensions() * sizeof(float);
    route.timing = measure([&](const auto& query) { return index.search(query, k); },
                           queries, warmups, repetitions);
    return route;
}

void write_route(std::ostream& out, const RouteSummary& route, bool last) {
    out << "    {\"route\": \"" << route.name << "\", \"backend\": \""
        << backend_name(route.backend) << "\", \"index_build_ms\": " << route.build_ms
        << ", \"index_storage_bytes\": " << route.index_storage_bytes
        << ", \"latency_median_ms\": " << route.timing.median_ms
        << ", \"latency_p95_ms\": " << route.timing.p95_ms
        << ", \"batch_queries_per_second\": " << route.timing.queries_per_second
        << ", \"queries_with_id_differences_vs_scalar\": "
        << route.queries_with_id_differences
        << ", \"different_neighbor_slots_vs_scalar\": "
        << route.different_neighbor_slots
        << ", \"max_ranked_distance_delta_vs_scalar\": "
        << route.max_ranked_distance_delta << "}" << (last ? "\n" : ",\n");
}

} // namespace

int main(int argc, char** argv) {
    try {
        const auto options = parse_options(argc, argv);
        const auto load_start = Clock::now();
        const auto database = cvs::load_vectors(options.database_path);
        const auto query_matrix = cvs::load_vectors(options.query_path);
        const auto load_end = Clock::now();
        if (database.rows() == 0 || query_matrix.rows() == 0 || options.k > database.rows()) {
            throw std::invalid_argument("need nonempty inputs and k <= database rows");
        }
        if (database.dimensions() != query_matrix.dimensions()) {
            throw std::invalid_argument("database and query dimension mismatch");
        }
        const auto queries = copy_queries(query_matrix);

        const cvs::ScalarFlatIndex scalar_index(database);
        auto scalar = measure_scalar(scalar_index, queries, options.k, options.warmup_queries,
                                    options.repetitions);
        const auto reference = collect_results(scalar_index, queries, options.k);
        auto blocked_scalar = measure_optimized(database, queries, options.k,
            options.warmup_queries, options.repetitions, cvs::ScoringBackend::scalar,
            "blocked_scalar");
        auto optimized = measure_optimized(database, queries, options.k,
            options.warmup_queries, options.repetitions, cvs::ScoringBackend::avx2,
            "blocked_optimized");
        const cvs::OptimizedFlatIndex blocked_scalar_index(database, cvs::ScoringBackend::scalar);
        const cvs::OptimizedFlatIndex optimized_index(database, cvs::ScoringBackend::avx2);
        compare_to_reference(blocked_scalar, collect_results(blocked_scalar_index, queries, options.k), reference);
        compare_to_reference(optimized, collect_results(optimized_index, queries, options.k), reference);

        std::ofstream file;
        if (!options.output_path.empty()) {
            file.open(options.output_path);
            if (!file) throw std::runtime_error("cannot create output: " + options.output_path);
        }
        std::ostream& out = options.output_path.empty() ? std::cout : file;
        out << "{\n  \"schema_version\": 1,\n  \"purpose\": \"optimized_flat_ablation\",\n"
            << "  \"operating_system\": \"" << operating_system() << "\",\n"
            << "  \"architecture\": \"" << architecture() << "\",\n"
            << "  \"compiler\": \"" << __VERSION__ << "\",\n"
            << "  \"build_type\": \"" << build_mode() << "\",\n"
            << "  \"machine_model\": \"" << json_escape(options.machine_model) << "\",\n"
            << "  \"data_seed\": \"" << json_escape(options.data_seed) << "\",\n"
            << "  \"database_file\": \"" << json_escape(options.database_path) << "\",\n"
            << "  \"query_file\": \"" << json_escape(options.query_path) << "\",\n"
            << "  \"database_rows\": " << database.rows() << ",\n"
            << "  \"query_rows\": " << query_matrix.rows() << ",\n"
            << "  \"dimensions\": " << database.dimensions() << ",\n"
            << "  \"k\": " << options.k << ",\n"
            << "  \"warmup_queries\": " << options.warmup_queries << ",\n"
            << "  \"repetitions\": " << options.repetitions << ",\n"
            << "  \"load_ms\": " << milliseconds(load_start, load_end) << ",\n"
            << "  \"database_bytes\": " << database.values().size() * sizeof(float) << ",\n"
            << "  \"routes\": [\n";
        write_route(out, scalar, false);
        write_route(out, blocked_scalar, false);
        write_route(out, optimized, true);
        out << "  ]\n}\n";
        out.flush();
        if (!out) throw std::runtime_error("failed to write ablation output");
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << '\n';
        return 1;
    }
}
