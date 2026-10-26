#include "cvs/benchmark.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <thread>
#include <unordered_set>

#if defined(__unix__) || defined(__APPLE__)
#include <sys/resource.h>
#endif
#if defined(__APPLE__)
#include <sys/sysctl.h>
#endif
#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <psapi.h>
#endif

namespace cvs::bench {

double milliseconds(Clock::time_point start, Clock::time_point end) {
    return std::chrono::duration<double, std::milli>(end - start).count();
}

double nearest_rank_percentile(const std::vector<double>& sorted, double p) {
    if (sorted.empty()) throw std::invalid_argument("percentile of an empty sample");
    if (!(p > 0.0 && p <= 1.0)) throw std::invalid_argument("percentile must be in (0, 1]");
    const auto rank = static_cast<std::size_t>(std::ceil(p * static_cast<double>(sorted.size())));
    return sorted[std::max<std::size_t>(1, rank) - 1];
}

namespace {

std::size_t parse_index(const std::string& text, const char* what, std::size_t line) {
    if (text.empty() || text.find_first_not_of("0123456789") != std::string::npos) {
        throw std::runtime_error(std::string("ground truth line ") + std::to_string(line) +
                                 ": " + what + " is not a non-negative integer");
    }
    const auto value = std::stoull(text);
    const auto converted = static_cast<std::size_t>(value);
    if (converted != value) throw std::runtime_error("ground truth value out of range");
    return converted;
}

std::vector<std::string> split_csv_line(const std::string& line) {
    std::vector<std::string> fields;
    std::string field;
    std::istringstream stream(line);
    while (std::getline(stream, field, ',')) fields.push_back(field);
    if (!line.empty() && line.back() == ',') fields.emplace_back();
    return fields;
}

} 

GroundTruth load_ground_truth_csv(const std::string& path) {
    std::ifstream input(path);
    if (!input) throw std::runtime_error("cannot open ground truth file: " + path);

    std::string line;
    if (!std::getline(input, line)) throw std::runtime_error("ground truth file is empty: " + path);
    if (!line.empty() && line.back() == '\r') line.pop_back();
    if (line != "query_id,rank,neighbor_id,squared_distance") {
        throw std::runtime_error("unexpected ground truth header in " + path);
    }

    GroundTruth truth;
    std::unordered_set<std::size_t> seen_in_query;
    std::size_t line_number = 1;
    while (std::getline(input, line)) {
        ++line_number;
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (line.empty()) continue;
        const auto fields = split_csv_line(line);
        if (fields.size() != 4) {
            throw std::runtime_error("ground truth line " + std::to_string(line_number) +
                                     " does not have 4 columns");
        }
        const auto query = parse_index(fields[0], "query_id", line_number);
        const auto rank = parse_index(fields[1], "rank", line_number);
        const auto neighbor = parse_index(fields[2], "neighbor_id", line_number);
        char* end = nullptr;
        const double distance = std::strtod(fields[3].c_str(), &end);
        if (fields[3].empty() || end != fields[3].c_str() + fields[3].size() ||
            !std::isfinite(distance) || distance < 0.0) {
            throw std::runtime_error("ground truth line " + std::to_string(line_number) +
                                     ": bad squared_distance");
        }

        if (rank == 1) {
            if (query != truth.ids.size()) {
                throw std::runtime_error("ground truth query IDs must start at 0 and be contiguous");
            }
            if (!truth.ids.empty()) {
                const auto previous = truth.ids.back().size();
                if (truth.k == 0) truth.k = previous;
                if (previous != truth.k) {
                    throw std::runtime_error("ground truth queries have different neighbour counts");
                }
            }
            truth.ids.emplace_back();
            seen_in_query.clear();
        } else if (truth.ids.empty() || query + 1 != truth.ids.size() ||
                   rank != truth.ids.back().size() + 1) {
            throw std::runtime_error("ground truth line " + std::to_string(line_number) +
                                     ": rank sequence is broken");
        }
        if (!seen_in_query.insert(neighbor).second) {
            throw std::runtime_error("ground truth repeats a neighbour within one query");
        }
        truth.ids.back().push_back(neighbor);
    }
    if (truth.ids.empty()) throw std::runtime_error("ground truth file has no answers: " + path);
    const auto last = truth.ids.back().size();
    if (truth.k == 0) truth.k = last;
    if (last != truth.k) {
        throw std::runtime_error("ground truth queries have different neighbour counts");
    }
    return truth;
}

double recall_at_k(const std::vector<std::size_t>& found_ids,
                   const std::vector<std::size_t>& expected, std::size_t k) {
    if (k == 0) throw std::invalid_argument("recall needs k > 0");
    if (expected.size() < k) {
        throw std::invalid_argument("reference has fewer than k neighbours for a query");
    }
    const std::unordered_set<std::size_t> reference(expected.begin(), expected.begin() + k);
    std::unordered_set<std::size_t> counted;
    const auto limit = std::min(k, found_ids.size());
    for (std::size_t i = 0; i < limit; ++i) {
        if (reference.count(found_ids[i]) != 0) counted.insert(found_ids[i]);
    }
    return static_cast<double>(counted.size()) / static_cast<double>(k);
}

double recall_at_k(const std::vector<Neighbor>& found,
                   const std::vector<std::size_t>& expected, std::size_t k) {
    std::vector<std::size_t> ids;
    ids.reserve(found.size());
    for (const auto& neighbor : found) ids.push_back(neighbor.id);
    return recall_at_k(ids, expected, k);
}

RecallSummary evaluate_recall(const std::vector<SearchResult>& results,
                              const GroundTruth& truth, std::size_t k) {
    if (results.size() != truth.ids.size()) {
        throw std::invalid_argument("ground truth has " + std::to_string(truth.ids.size()) +
                                    " queries but " + std::to_string(results.size()) +
                                    " were searched");
    }
    if (truth.k < k) throw std::invalid_argument("ground truth k is smaller than search k");
    RecallSummary summary;
    summary.per_query.reserve(results.size());
    summary.min = 1.0;
    double total = 0.0;
    for (std::size_t q = 0; q < results.size(); ++q) {
        const double value = recall_at_k(results[q].neighbors, truth.ids[q], k);
        summary.per_query.push_back(value);
        total += value;
        summary.min = std::min(summary.min, value);
        if (value == 1.0) ++summary.perfect_queries;
    }
    summary.mean = results.empty() ? 0.0 : total / static_cast<double>(results.size());
    return summary;
}

std::vector<std::vector<float>> copy_rows(const DenseMatrix& matrix) {
    std::vector<std::vector<float>> rows;
    rows.reserve(matrix.rows());
    for (std::size_t id = 0; id < matrix.rows(); ++id) {
        const float* row = matrix.row(id);
        rows.emplace_back(row, row + matrix.dimensions());
    }
    return rows;
}

DenseMatrix prefix_subset(const DenseMatrix& matrix, std::size_t rows) {
    if (rows > matrix.rows()) throw std::invalid_argument("subset is larger than the database");
    const auto end = matrix.values().begin() +
                     static_cast<std::ptrdiff_t>(rows * matrix.dimensions());
    return DenseMatrix(matrix.dimensions(), std::vector<float>(matrix.values().begin(), end));
}

LatencyRun measure_latency(const SearchIndex& index,
                           const std::vector<std::vector<float>>& queries,
                           std::size_t k, std::size_t warmup, std::size_t repetitions) {
    if (queries.empty()) throw std::invalid_argument("no queries to time");
    if (repetitions == 0) throw std::invalid_argument("repetitions must be positive");

    volatile double sink = 0.0;
    for (std::size_t q = 0; q < std::min(warmup, queries.size()); ++q) {
        const auto result = index.search(queries[q], k);
        for (const auto& n : result.neighbors) sink = sink + n.squared_distance;
    }

    LatencyRun run;
    run.records.reserve(queries.size() * repetitions);
    double candidate_total = 0.0;
    for (std::size_t rep = 0; rep < repetitions; ++rep) {
        for (std::size_t q = 0; q < queries.size(); ++q) {
            const auto start = Clock::now();
            const auto result = index.search(queries[q], k);
            const auto end = Clock::now();
            run.records.push_back({rep, q, milliseconds(start, end), result.candidates_examined});
            candidate_total += static_cast<double>(result.candidates_examined);
            for (const auto& n : result.neighbors) sink = sink + n.squared_distance;
        }
    }

    std::vector<double> sorted;
    sorted.reserve(run.records.size());
    for (const auto& record : run.records) sorted.push_back(record.latency_ms);
    std::sort(sorted.begin(), sorted.end());
    run.mean_ms = std::accumulate(sorted.begin(), sorted.end(), 0.0) /
                  static_cast<double>(sorted.size());
    run.median_ms = nearest_rank_percentile(sorted, 0.50);
    run.p95_ms = nearest_rank_percentile(sorted, 0.95);
    run.p99_ms = nearest_rank_percentile(sorted, 0.99);
    run.max_ms = sorted.back();
    run.mean_candidates = candidate_total / static_cast<double>(run.records.size());
    return run;
}

namespace {
#if defined(__linux__)
std::optional<std::size_t> status_field_bytes(const char* name) {
    std::ifstream status("/proc/self/status");
    std::string key;
    while (status >> key) {
        if (key == name) {
            std::size_t kilobytes = 0;
            if (status >> kilobytes) return kilobytes * 1024;
            return std::nullopt;
        }
        status.ignore(4096, '\n');
    }
    return std::nullopt;
}
#endif
}

MemorySnapshot read_memory() {
    MemorySnapshot snapshot;
#if defined(__linux__)
    snapshot.current_rss_bytes = status_field_bytes("VmRSS:");
    snapshot.peak_rss_bytes = status_field_bytes("VmHWM:");
#elif defined(__APPLE__)
    rusage usage{};
    if (getrusage(RUSAGE_SELF, &usage) == 0) {
        snapshot.peak_rss_bytes = static_cast<std::size_t>(usage.ru_maxrss);
    }
#elif defined(_WIN32)
    PROCESS_MEMORY_COUNTERS counters{};
    if (GetProcessMemoryInfo(GetCurrentProcess(), &counters, sizeof(counters))) {
        snapshot.current_rss_bytes = static_cast<std::size_t>(counters.WorkingSetSize);
        snapshot.peak_rss_bytes = static_cast<std::size_t>(counters.PeakWorkingSetSize);
    }
#endif
    return snapshot;
}

bool reset_peak_memory() {
#if defined(__linux__)
    std::ofstream clear("/proc/self/clear_refs");
    if (!clear) return false;
    clear << "5";
    clear.flush();
    if (!clear) return false;
    const auto snapshot = read_memory();
    return snapshot.current_rss_bytes && snapshot.peak_rss_bytes &&
           *snapshot.peak_rss_bytes <= *snapshot.current_rss_bytes + 64 * 1024;
#else
    return false;
#endif
}

const char* memory_method() {
#if defined(__linux__)
    return "linux /proc/self/status VmRSS and VmHWM (resident set size); "
           "VmHWM reset through /proc/self/clear_refs after loading when the kernel allows it";
#elif defined(__APPLE__)
    return "macOS getrusage ru_maxrss (whole-process peak resident set size)";
#elif defined(_WIN32)
    return "windows GetProcessMemoryInfo WorkingSetSize and PeakWorkingSetSize "
           "(whole-process peak; cannot be reset after loading)";
#else
    return "unsupported platform";
#endif
}

std::string cpu_model() {
#if defined(__linux__)
    std::ifstream info("/proc/cpuinfo");
    std::string line;
    while (std::getline(info, line)) {
        for (const char* key : {"model name", "Model", "Hardware"}) {
            if (line.rfind(key, 0) == 0) {
                const auto colon = line.find(':');
                if (colon != std::string::npos) {
                    auto value = line.substr(colon + 1);
                    value.erase(0, value.find_first_not_of(" \t"));
                    if (!value.empty()) return value;
                }
            }
        }
    }
#elif defined(__APPLE__)
    char buffer[256] = {};
    std::size_t size = sizeof(buffer);
    if (sysctlbyname("machdep.cpu.brand_string", buffer, &size, nullptr, 0) == 0) return buffer;
#elif defined(_WIN32)
    char buffer[256] = {};
    DWORD size = sizeof(buffer);
    if (RegGetValueA(HKEY_LOCAL_MACHINE, "HARDWARE\\DESCRIPTION\\System\\CentralProcessor\\0",
                     "ProcessorNameString", RRF_RT_REG_SZ, nullptr, buffer, &size) == ERROR_SUCCESS) {
        std::string value(buffer);
        value.erase(0, value.find_first_not_of(" \t"));
        while (!value.empty() && value.back() == ' ') value.pop_back();
        if (!value.empty()) return value;
    }
#endif
    return "unknown";
}

std::size_t hardware_threads() {
    return static_cast<std::size_t>(std::thread::hardware_concurrency());
}

} 