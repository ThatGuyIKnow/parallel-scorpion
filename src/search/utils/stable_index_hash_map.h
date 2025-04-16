#ifndef STABLE_INDEX_MAP_H
#define STABLE_INDEX_MAP_H

#include "hash.h"

#include <parallel_hashmap/phmap.h>

#include <vector>

namespace utils {
class StableIndexMap {
    unsigned int probes = 0;
    unsigned int calls = 0;
    unsigned int misses = 0;

    struct Entry {
        int left = -1;
        int right = -1;

        bool operator==(const Entry &rhs) const {
            return left == rhs.left && right == rhs.right;
        }

        bool operator==(const std::vector<int> &rhs) const {
            if (rhs.size() == 1) {
                return left == rhs[0];
            }
            return left == rhs[0] && right == rhs[1];
        }

        friend size_t hash_value(const Entry &entry) {
            utils::HashState hash_state;
            hash_state.feed(entry.left);
            hash_state.feed(entry.right);
            return hash_state.get_hash64();
        }
    };

    using StateSet = phmap::flat_hash_map<Entry, int>;

    StateSet _values;
    int _size = 0;

public:
    // Insert operations
    int insert(const std::vector<int> &vec);

    int find_or_insert(const std::vector<int> &vec);

    // Access operations
    bool contains(const std::vector<int> &vec);
    int find(const std::vector<int> &vec);

    void print_info();

    size_t size();
};
}
#endif // STABLE_INDEX_MAP_H
