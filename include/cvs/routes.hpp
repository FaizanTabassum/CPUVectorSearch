#pragma once

#include "cvs/search.hpp"

#include <map>
#include <memory>
#include <set>
#include <string>
#include <vector>

namespace cvs::bench {

class RouteParams {
public:
    RouteParams() = default;
    explicit RouteParams(std::map<std::string, std::string> values);

    void add(const std::string& assignment);

    std::string text(const std::string& key, const std::string& fallback);
    std::size_t positive_integer(const std::string& key, std::size_t fallback);
    bool has(const std::string& key) const;

    void require_all_used() const;
    const std::map<std::string, std::string>& values() const noexcept { return values_; }

private:
    std::map<std::string, std::string> values_;
    std::set<std::string> used_;
};

struct BuiltRoute {
    std::unique_ptr<SearchIndex> index;
    std::map<std::string, std::string> metadata;
};

std::vector<std::string> available_routes();

BuiltRoute build_route(const std::string& name, const DenseMatrix& database, RouteParams params);

}
