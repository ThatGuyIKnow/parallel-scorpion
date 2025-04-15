

#include "stable_index_hash_map.h"

#include <algorithm>

#include "logging.h"
#include "treedbs.h"


namespace utils {
    StableIndexMap::StableIndexMap(size_t initial_size, float load_factor, size_t max_size) :
        initial_size(initial_size), load_factor(load_factor), max_size(max_size) {
    }

    
    int StableIndexMap::insert(const std::vector<int>& value) {
        assert(value.size() <= 2 && !value.empty());
        Entry entry = {value[0], value.size() < 2 ? -1 : value[1]};

        _values.insert({entry, _size});
        return _size++;
    }

    
    int StableIndexMap::find_or_insert(const std::vector<int>& value) {
        Entry entry = {value[0], value.size() < 2 ? -1 : value[1]};
        auto it = _values.find(entry);
        if (it != _values.end()) {
            return it->second;
        }

        return -1;
    }

    
    bool StableIndexMap::contains(const std::vector<int>& vec) {
        assert(vec.size() <= 2 && !vec.empty());

        Entry entry = {vec[0], vec.size() < 2 ? -1 : vec[1]};
        return _values.contains(entry);
    }


    
    int StableIndexMap::find(const std::vector<int>& vec) {
        assert(vec.size() <= 2 && !vec.empty());

        Entry entry = {vec[0], vec.size() < 2 ? -1 : vec[1]};

        if (!_values.contains(entry))
            return -1;
        return _values.find(entry)->second;
    }


    
    size_t StableIndexMap::size() {
        return _size;
    }

    void StableIndexMap::print_info() {
        g_log << "Number of stable index map entries: " << size()
            << "\nLoad factor: " << load_factor
            << "\nMax size: " << max_size
            << "\nNumber of probes: " << probes
            << "\nNumber of calls: " << calls
            << "\nNumber of misses: " << std::endl;
    }




}
