#ifndef UTILS_TREEDBS_H
#define UTILS_TREEDBS_H

#include "hash.h"
#include "stable_index_hash_map.h"

#include "../algorithms/int_packer.h"
#include "../algorithms/segmented_vector.h"

#include <cstddef>
#include <vector>

namespace utils {
class TreeDBS {
    StableIndexMap entries;
    size_t _size;

    // Iterative helper functions
    int putRecursively(const std::vector<int>::const_iterator begin, const std::vector<int>::const_iterator end, int index);
    int findRecursively(std::vector<int>::const_iterator begin, std::vector<int>::const_iterator end, int index);

    int put(std::vector<int>::const_iterator begin, std::vector<int>::const_iterator end);

    size_t split_range(size_t start, size_t end) const {
        auto diff = end - start;
        return diff / 2 + diff % 2 + start;
    }

public:
    explicit TreeDBS(size_t size);

    void insert(const std::vector<int> &vec);
    bool contains(const std::vector<int> &vec);
    size_t size();

    int find(std::vector<int>::const_iterator begin, std::vector<int>::const_iterator end);

    void print_info();
};
}

#endif
