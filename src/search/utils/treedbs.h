#ifndef TREEDBS_HPP
#define TREEDBS_HPP

#include <vector>
#include <unordered_map>
#include <memory>
#include <cstddef>
#include <utility>
#include <stack>


#include "../algorithms/int_packer.h"
#include "../algorithms/segmented_vector.h"
#include "hash.h"

#include "stable_index_hash_map.h"

namespace int_packer {
    class IntPacker;
}

using PackedStateBin = int_packer::IntPacker::Bin;


namespace utils {

class TreeDBS {
    struct Node {
        std::unique_ptr<Node> left;
        std::unique_ptr<Node> right;
        int level = 0;
        bool isLeaf = false;
        StableIndexMap entries;
    };

    std::unique_ptr<Node> root;
    size_t _size;
    std::unique_ptr<Node> constructTreeHelper(size_t input_size, int level = 0, StableIndexMap entries = StableIndexMap());

    // Iterative helper functions
    int putRecursively(std::vector<int>::const_iterator begin, std::vector<int>::const_iterator end, const std::unique_ptr<Node>& node);
    int findRecursively(std::vector<int>::const_iterator begin, std::vector<int>::const_iterator end, const std::unique_ptr<Node>& node);


    size_t splitRange(size_t start, size_t end) const {
        auto diff = end - start;
        return diff / 2 + diff % 2 + start;
    }

public:
    explicit TreeDBS(size_t size);

    void insert(const std::vector<int>& vec);
    bool contains(const std::vector<int>& vec);
    size_t size() const;
    void print_info() const;
};
}
#endif // TREEDBS_HPP
