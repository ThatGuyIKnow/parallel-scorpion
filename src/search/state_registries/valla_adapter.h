#ifndef VALLA_ADAPTER_H
#define VALLA_ADAPTER_H

#include <unordered_map>
#include <vector>

// Custom hash function for vectors
struct VectorHash {
    template<typename T>
    std::size_t operator()(const std::vector<T>& vec) const {
        std::size_t seed = vec.size();
        for (const auto& elem : vec) {
            seed ^= std::hash<T>{}(elem) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
        }
        return seed;
    }
};

// Simple adapter for valla migration - use standard hash map with proper hash function
namespace vs {
    template<typename T>
    class IndexedHashSet {
    private:
        std::unordered_map<T, uint32_t, VectorHash> hash_to_index;
        std::vector<T> indexed_values;
        uint32_t next_index = 0;

    public:
        uint32_t insert(const T& value) {
            auto it = hash_to_index.find(value);
            if (it != hash_to_index.end()) {
                return it->second;
            }

            uint32_t index = next_index++;
            hash_to_index[value] = index;
            indexed_values.push_back(value);
            return index;
        }

        T lookup(uint32_t index) const {
            return indexed_values[index];
        }

        size_t size() const {
            return indexed_values.size();
        }

        size_t mem_usage() const {
            return indexed_values.size() * sizeof(T) + hash_to_index.size() * (sizeof(T) + sizeof(uint32_t));
        }
    };
}

#endif
