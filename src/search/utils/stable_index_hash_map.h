#ifndef STABLE_INDEX_MAP_H
#define STABLE_INDEX_MAP_H

#include <parallel_hashmap/phmap.h>
#include <vector>
#include <concepts>
#include <functional>
#include <stdexcept>
#include <utility>
#include <pstl/glue_execution_defs.h>
#include <execution>

#include "hash.h"

template<typename T>
constexpr typename std::vector<T>::iterator circular_find(typename std::vector<T>::iterator start,
                                                          typename std::vector<T>::iterator end,
                                                          const T& value, size_t index) {
    // First search from index to end
    auto it = std::find(std::execution::unseq, start + index, end, value);

    // If not found, search from beginning to index
    if (it == end && index > 0) {
        it = std::find(std::execution::unseq, start, start + index, value);
        it = it == start + index ? end : it;
    }

    return it;
}

template<typename T>
constexpr typename std::vector<T>::iterator circular_find_first_of(typename std::vector<T>::iterator start,
                                                                   typename std::vector<T>::iterator end,
                                                                   typename std::vector<T>::iterator value_start,
                                                                   typename std::vector<T>::iterator value_end,
                                                                   size_t index) {
    // First search from index to end
    auto it = std::find_first_of(std::execution::unseq, start + index, end, value_start, value_end);

    // If not found, search from beginning to index
    if (it == end && index > 0) {
        it = std::find_first_of(std::execution::unseq, start, start + index, value_start, value_end);
        it = it == start + index ? end : it;
    }

    return it;
}

namespace utils {
    class StableIndexMap {


        unsigned int probes = 0;
        unsigned int calls = 0;
        unsigned int misses = 0;

        struct Entry {
            int left = -1;
            int right = -1;

            bool operator==(const Entry& rhs) const {
                return left == rhs.left && right == rhs.right;
            }

            bool operator==(const std::vector<int>& rhs) const {
                if (rhs.size() == 1) {
                    return left == rhs[0];
                }
                return left == rhs[0] && right == rhs[1];
            }

            friend size_t hash_value(const Entry &entry)
            {
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
        explicit StableIndexMap();

        // Insert operations
        int insert(const std::vector<int>& vec);

        int find_or_insert(const std::vector<int>& vec);

        // Access operations
        bool contains(const std::vector<int>& vec);
        int find(const std::vector<int>& vec);

        void print_info();

        size_t size();
    };
}
#endif // STABLE_INDEX_MAP_H