#include "cvs/routes.hpp"

#include "cvs/ivf.hpp"
#include "cvs/kmeans.hpp"
#include "cvs/optimized_flat.hpp"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <utility>

namespace cvs::bench {

RouteParams::RouteParams(std::map<std::string, std::string> values)
    : values_(std::move(values)) {}

void RouteParams::add(const std::string& assignment) {
    const auto equals = assignment.find('=');
    if (equals == std::string::npos || equals == 0) {
        throw std::invalid_argument("route parameter must look like key=value: " + assignment);
    }
    const auto key = assignment.substr(0, equals);
    if (!values_.emplace(key, assignment.substr(equals + 1)).second) {
        throw std::invalid_argument("route parameter given twice: " + key);
    }
}

bool RouteParams::has(const std::string& key) const { return values_.count(key) != 0; }

std::string RouteParams::text(const std::string& key, const std::string& fallback) {
    const auto found = values_.find(key);
    if (found == values_.end()) return fallback;
    used_.insert(key);
    return found->second;
}

std::size_t RouteParams::positive_integer(const std::string& key, std::size_t fallback) {
    const auto found = values_.find(key);
    if (found == values_.end()) return fallback;
    used_.insert(key);
    const auto& value = found->second;
    if (value.empty() || value.find_first_not_of("0123456789") != std::string::npos) {
        throw std::invalid_argument("route parameter " + key + " must be a positive integer");
    }
    const auto parsed = std::stoull(value);
    const auto converted = static_cast<std::size_t>(parsed);
    if (converted == 0 || converted != parsed) {
        throw std::invalid_argument("route parameter " + key + " is out of range");
    }
    return converted;
}

void RouteParams::require_all_used() const {
    for (const auto& entry : values_) {
        if (used_.count(entry.first) == 0) {
            throw std::invalid_argument("route does not understand parameter: " + entry.first);
        }
    }
}

namespace {

const char* backend_name(ScoringBackend backend) {
    switch (backend) {
    case ScoringBackend::automatic: return "automatic";
    case ScoringBackend::scalar: return "scalar";
    case ScoringBackend::avx2: return "avx2";
    }
    return "unknown";
}

ScoringBackend parse_backend(const std::string& text) {
    if (text == "auto" || text == "automatic") return ScoringBackend::automatic;
    if (text == "scalar") return ScoringBackend::scalar;
    if (text == "avx2") return ScoringBackend::avx2;
    throw std::invalid_argument("backend must be auto, scalar or avx2");
}

BuiltRoute scalar_flat(const DenseMatrix& database, RouteParams& params) {
    params.require_all_used();
    BuiltRoute route;
    route.index = std::make_unique<ScalarFlatIndex>(database);
    route.metadata["backend"] = "scalar";
    return route;
}

BuiltRoute optimized_flat(const DenseMatrix& database, RouteParams& params) {
    const auto requested = parse_backend(params.text("backend", "auto"));
    params.require_all_used();
    auto index = std::make_unique<OptimizedFlatIndex>(database, requested);
    BuiltRoute route;
    route.metadata["backend_requested"] = backend_name(requested);
    route.metadata["backend"] = backend_name(index->backend());
    route.metadata["padded_dimensions"] = std::to_string(index->padded_dimensions());
    route.metadata["padded_index_bytes"] =
        std::to_string(database.rows() * index->padded_dimensions() * sizeof(float));
    route.index = std::move(index);
    return route;
}

BuiltRoute ivf(const DenseMatrix& database, RouteParams& params) {
    const auto lists = params.positive_integer("lists", 64);
    const auto probes = params.positive_integer("probes", 8);
    const auto seed = params.positive_integer("seed", 5206);
    const auto iterations = params.positive_integer("iterations", 25);
    const auto requested = parse_backend(params.text("backend", "auto"));
    params.require_all_used();
    if (lists > database.rows()) {
        throw std::invalid_argument("ivf needs lists <= database rows");
    }
    if (probes > lists) throw std::invalid_argument("ivf needs probes <= lists");
    if (seed > std::numeric_limits<std::uint32_t>::max()) {
        throw std::invalid_argument("ivf seed must fit in 32 bits");
    }

    auto centroids = train_kmeans(database, lists, static_cast<std::uint32_t>(seed), iterations);
    auto index = std::make_unique<IvfIndex>(database, std::move(centroids), probes, requested);

    std::size_t empty_lists = 0;
    std::size_t largest_list = 0;
    for (std::size_t id = 0; id < index->nlist(); ++id) {
        const auto size = index->list(id).size();
        if (size == 0) ++empty_lists;
        largest_list = std::max(largest_list, size);
    }

    BuiltRoute route;
    route.metadata["lists"] = std::to_string(lists);
    route.metadata["probes"] = std::to_string(probes);
    route.metadata["seed"] = std::to_string(seed);
    route.metadata["iterations"] = std::to_string(iterations);
    route.metadata["backend_requested"] = backend_name(requested);
    route.metadata["backend"] = backend_name(index->backend());
    route.metadata["empty_lists"] = std::to_string(empty_lists);
    route.metadata["largest_list"] = std::to_string(largest_list);
    route.index = std::move(index);
    return route;
}

}

std::vector<std::string> available_routes() {
    return {"scalar_flat", "optimized_flat", "ivf"};
}

BuiltRoute build_route(const std::string& name, const DenseMatrix& database, RouteParams params) {
    if (name == "scalar_flat") return scalar_flat(database, params);
    if (name == "optimized_flat") return optimized_flat(database, params);
    if (name == "ivf") return ivf(database, params);
    throw std::invalid_argument("unknown route: " + name + " (try --list-routes)");
}

}
