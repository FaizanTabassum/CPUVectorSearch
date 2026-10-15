#include "cvs/io.hpp"
#include "cvs/search.hpp"

#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {
std::size_t positive_integer(const std::string& text) {
    if (text.empty() || text.find_first_not_of("0123456789") != std::string::npos)
        throw std::invalid_argument("expected a positive integer");
    const auto value = std::stoull(text);
    const auto converted = static_cast<std::size_t>(value);
    if (value == 0 || converted != value) throw std::invalid_argument("integer out of range");
    return converted;
}
}

int main(int argc, char** argv) {
    try {
        std::string database_path, query_path, output_path;
        std::size_t database_size = 0, k = 10;
        for (int i = 1; i < argc; ++i) {
            const std::string option = argv[i];
            if (option == "--help") {
                std::cout << "Usage: reference_search --database FILE --queries FILE "
                             "--database-size N --output CSV [--k K]\n"
                             "IDs are zero-based row positions; rank starts at 1.\n";
                return 0;
            }
            if (i + 1 >= argc) throw std::invalid_argument("missing value for " + option);
            const std::string value = argv[++i];
            if (option == "--database") database_path = value;
            else if (option == "--queries") query_path = value;
            else if (option == "--output") output_path = value;
            else if (option == "--database-size") database_size = positive_integer(value);
            else if (option == "--k") k = positive_integer(value);
            else throw std::invalid_argument("unknown option: " + option);
        }
        if (database_path.empty() || query_path.empty() || output_path.empty() || database_size == 0)
            throw std::invalid_argument("database, queries, database-size and output are required");
        if (std::filesystem::exists(output_path))
            throw std::invalid_argument("output already exists; choose a new path");
        const auto full_database = cvs::load_vectors(database_path);
        const auto queries = cvs::load_vectors(query_path);
        if (database_size > full_database.rows() || k > database_size || queries.rows() == 0)
            throw std::invalid_argument("need 0 < k <= subset size <= database rows and nonempty queries");
        if (full_database.dimensions() != queries.dimensions())
            throw std::invalid_argument("database and query dimension mismatch");

        // IDs remain the original row positions in each nested prefix subset.
        const auto end = full_database.values().begin() + database_size * full_database.dimensions();
        const cvs::DenseMatrix database(full_database.dimensions(),
                                       std::vector<float>(full_database.values().begin(), end));
        const cvs::ScalarFlatIndex index(database);
        std::ofstream output(output_path);
        if (!output) throw std::runtime_error("cannot create output file");
        output << "query_id,rank,neighbor_id,squared_distance\n"
               << std::setprecision(std::numeric_limits<float>::max_digits10);
        for (std::size_t q = 0; q < queries.rows(); ++q) {
            const auto* row = queries.row(q);
            const std::vector<float> query(row, row + queries.dimensions());
            const auto result = index.search(query, k);
            for (std::size_t rank = 0; rank < result.neighbors.size(); ++rank) {
                const auto& neighbor = result.neighbors[rank];
                if (!std::isfinite(neighbor.squared_distance))
                    throw std::runtime_error("non-finite distance; check input vector scales");
                output << q << ',' << rank + 1 << ',' << neighbor.id << ','
                       << neighbor.squared_distance << '\n';
            }
        }
        output.close();
        if (!output) throw std::runtime_error("failed to finish output file");
        std::cout << "Saved " << queries.rows() << " queries x " << k
                  << " neighbors; database size=" << database_size << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << '\n';
        return 1;
    }
}
