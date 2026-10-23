#include "cvs/benchmark.hpp"
#include "cvs/io.hpp"
#include "cvs/routes.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using cvs::bench::Clock;
using cvs::bench::milliseconds;

struct Options {
    std::string database_path;
    std::string query_path;
    std::string output_path;
    std::string raw_output_path;
    std::string results_output_path;
    std::string ground_truth_path;
    std::string route = "scalar_flat";
    std::string label;
    cvs::bench::RouteParams params;
    std::map<std::string, std::string> metadata;
    std::size_t database_size = 0;
    std::size_t k = 10;
    std::size_t warmup_queries = 10;
    std::size_t repetitions = 1;
};

const char* kUsage =
    "Usage: vector_search --database FILE --queries FILE [options]\n"
    "\n"
    "  --database-size N        use only the first N database rows (IDs unchanged)\n"
    "  --ground-truth CSV       Member A's answer sheet for this subset; enables recall\n"
    "  --k N                    neighbours per query (default 10)\n"
    "  --route NAME             scalar_flat (default), optimized_flat or ivf\n"
    "  --param KEY=VALUE        route setting, repeatable (backend=scalar, lists=64, probes=8)\n"
    "  --list-routes            print the routes this build knows and exit\n"
    "  --warmup-queries N       untimed searches before latency timing (default 10)\n"
    "  --repetitions N          timed passes over all queries (default 1)\n"
    "  --output FILE            summary JSON (stdout if omitted)\n"
    "  --raw-output FILE        per-query CSV: repetition, query, latency, candidates, recall\n"
    "  --results-output FILE    returned neighbours in the answer-sheet CSV format\n"
    "  --label TEXT             free-text label stored in the JSON\n"
    "  --metadata KEY=VALUE     extra provenance stored in the JSON, repeatable\n";

std::size_t parse_count(const std::string& value, const char* name, bool allow_zero = false) {
    if (value.empty() || value.find_first_not_of("0123456789") != std::string::npos) {
        throw std::invalid_argument(std::string(name) + " must be a positive integer");
    }
    const auto parsed = std::stoull(value);
    const auto converted = static_cast<std::size_t>(parsed);
    if (converted != parsed || (!allow_zero && converted == 0)) {
        throw std::invalid_argument(std::string(name) + " out of range");
    }
    return converted;
}

bool parse_options(int argc, char** argv, Options& options) {
    for (int i = 1; i < argc; ++i) {
        const std::string option = argv[i];
        if (option == "--help") {
            std::cout << kUsage;
            return false;
        }
        if (option == "--list-routes") {
            for (const auto& name : cvs::bench::available_routes()) std::cout << name << '\n';
            return false;
        }
        if (i + 1 >= argc) throw std::invalid_argument("missing value for " + option);
        const std::string value = argv[++i];
        if (option == "--database") options.database_path = value;
        else if (option == "--queries") options.query_path = value;
        else if (option == "--output") options.output_path = value;
        else if (option == "--raw-output") options.raw_output_path = value;
        else if (option == "--results-output") options.results_output_path = value;
        else if (option == "--route") options.route = value;
        else if (option == "--param") options.params.add(value);
        else if (option == "--ground-truth") options.ground_truth_path = value;
        else if (option == "--label") options.label = value;
        else if (option == "--metadata") {
            const auto equals = value.find('=');
            if (equals == std::string::npos || equals == 0)
                throw std::invalid_argument("--metadata must look like key=value");
            options.metadata[value.substr(0, equals)] = value.substr(equals + 1);
        }
        else if (option == "--database-size") options.database_size = parse_count(value, "database size");
        else if (option == "--k") options.k = parse_count(value, "k");
        else if (option == "--warmup-queries") options.warmup_queries = parse_count(value, "warmup", true);
        else if (option == "--repetitions") options.repetitions = parse_count(value, "repetitions");
        else throw std::invalid_argument("unknown option: " + option);
    }
    if (options.database_path.empty() || options.query_path.empty()) {
        throw std::invalid_argument("--database and --queries are required; see --help");
    }
    return true;
}

std::string escape(const std::string& text) {
    std::ostringstream out;
    for (const unsigned char c : text) {
        switch (c) {
        case '"': out << "\\\""; break;
        case '\\': out << "\\\\"; break;
        case '\n': out << "\\n"; break;
        case '\r': out << "\\r"; break;
        case '\t': out << "\\t"; break;
        default:
            if (c < 0x20) out << "\\u" << std::hex << std::setw(4) << std::setfill('0') << int(c)
                              << std::dec << std::setfill(' ');
            else out << c;
        }
    }
    return out.str();
}

std::string json_string(const std::string& text) { return "\"" + escape(text) + "\""; }

std::string number(double value) {
    if (!std::isfinite(value)) return "null";
    std::ostringstream out;
    out << std::setprecision(10) << value;
    return out.str();
}

std::string string_map(const std::map<std::string, std::string>& values) {
    std::string out = "{";
    bool first = true;
    for (const auto& entry : values) {
        out += (first ? "" : ", ") + json_string(entry.first) + ": " + json_string(entry.second);
        first = false;
    }
    return out + "}";
}

std::ofstream open_output(const std::string& path) {
    std::ofstream file(path);
    if (!file) throw std::runtime_error("cannot create output: " + path);
    return file;
}

}

int main(int argc, char** argv) {
    try {
        Options options;
        if (!parse_options(argc, argv, options)) return 0;

        const auto load_start = Clock::now();
        std::size_t file_rows = 0;
        const cvs::DenseMatrix database = [&] {
            auto full = cvs::load_vectors(options.database_path);
            file_rows = full.rows();
            if (options.database_size == 0 || options.database_size == full.rows()) return full;
            if (options.database_size > full.rows())
                throw std::invalid_argument("--database-size is larger than the database file");
            return cvs::bench::prefix_subset(full, options.database_size);
        }();
        const auto query_matrix = cvs::load_vectors(options.query_path);
        const auto load_end = Clock::now();

        if (database.rows() == 0 || query_matrix.rows() == 0 || options.k > database.rows())
            throw std::invalid_argument("need nonempty inputs and k <= database rows");
        if (database.dimensions() != query_matrix.dimensions())
            throw std::invalid_argument("database and query dimension mismatch");

        std::optional<cvs::bench::GroundTruth> truth;
        if (!options.ground_truth_path.empty()) {
            truth = cvs::bench::load_ground_truth_csv(options.ground_truth_path);
            if (truth->ids.size() != query_matrix.rows())
                throw std::invalid_argument("ground truth query count does not match the query file");
            if (truth->k < options.k)
                throw std::invalid_argument("ground truth has fewer neighbours per query than k");
            for (const auto& row : truth->ids)
                for (const auto id : row)
                    if (id >= database.rows())
                        throw std::invalid_argument(
                            "ground truth mentions an ID outside this database subset; "
                            "is it the answer sheet for a different size?");
        }
        const auto queries = cvs::bench::copy_rows(query_matrix);

        const auto build_start = Clock::now();
        auto route = cvs::bench::build_route(options.route, database, options.params);
        const auto build_end = Clock::now();
        const cvs::SearchIndex& index = *route.index;

        const auto latency = cvs::bench::measure_latency(
            index, queries, options.k, options.warmup_queries, options.repetitions);

        std::vector<cvs::SearchResult> results;
        results.reserve(queries.size());
        for (const auto& query : queries) results.push_back(index.search(query, options.k));
        std::optional<cvs::bench::RecallSummary> recall;
        if (truth) recall = cvs::bench::evaluate_recall(results, *truth, options.k);

        volatile double sink = 0.0;
        const auto batch_start = Clock::now();
        for (const auto& query : queries) {
            const auto result = index.search(query, options.k);
            for (const auto& n : result.neighbors) sink = sink + n.squared_distance;
        }
        const auto batch_end = Clock::now();
        const double batch_ms = milliseconds(batch_start, batch_end);
        if (batch_ms <= 0) throw std::runtime_error("timer resolution too coarse for batch");

        double checksum = 0.0;
        for (const auto& result : results)
            for (const auto& n : result.neighbors)
                checksum += static_cast<double>(n.id) + n.squared_distance;

        if (!options.raw_output_path.empty()) {
            auto raw = open_output(options.raw_output_path);
            raw << "repetition,query_id,latency_ms,candidates_examined,recall_at_k\n"
                << std::setprecision(9);
            for (const auto& record : latency.records) {
                raw << record.repetition << ',' << record.query_id << ',' << record.latency_ms
                    << ',' << record.candidates_examined << ',';
                if (recall) raw << recall->per_query[record.query_id];
                raw << '\n';
            }
            if (!raw) throw std::runtime_error("failed to write raw output");
        }

        if (!options.results_output_path.empty()) {
            auto found = open_output(options.results_output_path);
            found << "query_id,rank,neighbor_id,squared_distance\n"
                  << std::setprecision(std::numeric_limits<float>::max_digits10);
            for (std::size_t q = 0; q < results.size(); ++q) {
                const auto& neighbors = results[q].neighbors;
                for (std::size_t r = 0; r < neighbors.size(); ++r)
                    found << q << ',' << r + 1 << ',' << neighbors[r].id << ','
                          << neighbors[r].squared_distance << '\n';
            }
            if (!found) throw std::runtime_error("failed to write results output");
        }

        std::ofstream file;
        if (!options.output_path.empty()) file = open_output(options.output_path);
        std::ostream& out = options.output_path.empty() ? std::cout : file;
        out << "{\n"
            << "  \"schema_version\": 2,\n"
            << "  \"purpose\": \"benchmark\",\n"
            << "  \"label\": " << json_string(options.label) << ",\n"
            << "  \"route\": " << json_string(options.route) << ",\n"
            << "  \"route_parameters\": " << string_map(options.params.values()) << ",\n"
            << "  \"route_metadata\": " << string_map(route.metadata) << ",\n"
            << "  \"database_file\": " << json_string(options.database_path) << ",\n"
            << "  \"query_file\": " << json_string(options.query_path) << ",\n"
            << "  \"ground_truth_file\": "
            << (truth ? json_string(options.ground_truth_path) : "null") << ",\n"
            << "  \"database_file_rows\": " << file_rows << ",\n"
            << "  \"database_rows\": " << database.rows() << ",\n"
            << "  \"query_rows\": " << query_matrix.rows() << ",\n"
            << "  \"dimensions\": " << database.dimensions() << ",\n"
            << "  \"k\": " << options.k << ",\n"
            << "  \"threads\": 1,\n"
            << "  \"warmup_queries\": " << std::min(options.warmup_queries, queries.size()) << ",\n"
            << "  \"repetitions\": " << options.repetitions << ",\n"
            << "  \"load_ms\": " << number(milliseconds(load_start, load_end)) << ",\n"
            << "  \"index_build_ms\": " << number(milliseconds(build_start, build_end)) << ",\n"
            << "  \"latency_mean_ms\": " << number(latency.mean_ms) << ",\n"
            << "  \"latency_median_ms\": " << number(latency.median_ms) << ",\n"
            << "  \"latency_p95_ms\": " << number(latency.p95_ms) << ",\n"
            << "  \"latency_p99_ms\": " << number(latency.p99_ms) << ",\n"
            << "  \"latency_max_ms\": " << number(latency.max_ms) << ",\n"
            << "  \"batch_queries_per_second\": "
            << number(1000.0 * static_cast<double>(queries.size()) / batch_ms) << ",\n"
            << "  \"candidates_per_query\": " << number(latency.mean_candidates) << ",\n"
            << "  \"recall_at_k\": " << (recall ? number(recall->mean) : "null") << ",\n"
            << "  \"recall_min\": " << (recall ? number(recall->min) : "null") << ",\n"
            << "  \"recall_perfect_queries\": "
            << (recall ? std::to_string(recall->perfect_queries) : "null") << ",\n"
            << "  \"database_vector_bytes\": " << database.values().size() * sizeof(float) << ",\n"
            << "  \"peak_memory_bytes\": null,\n"
            << "  \"metadata\": " << string_map(options.metadata) << ",\n"
            << "  \"checksum\": " << number(checksum) << "\n"
            << "}\n";
        out.flush();
        if (!out) throw std::runtime_error("failed to write benchmark output");
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << '\n';
        return 1;
    }
}
