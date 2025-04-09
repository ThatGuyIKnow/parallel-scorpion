

#include "stable_index_hash_map.h"

#include <algorithm>

#include "logging.h"
#include "treedbs.h"


namespace utils {


    
    StableIndexMap::StableIndexMap(size_t initial_size, float load_factor, size_t max_size) :
        initial_size(initial_size), load_factor(load_factor), max_size(max_size) {
        _resize(initial_size);
    }

    
    int StableIndexMap::insert(const std::vector<int>& value) {
        assert (std::find(_values->begin(), _values->end(), value) == _values->end());
        assert(value.size() <= 2 && !value.empty());

        auto index = _hasher.hashVector(value) % _values->size();
        typename std::vector<Entry>::iterator it = circular_find<Entry>(_values->begin(), _values->end(), {}, index);

        if (it == _values->end()) {
            return _values->size();
        }

        auto new_index = std::distance(_values->begin(), it);

        _values->operator[](new_index) = {value[0], value.size() < 2 ? -1 : value[1]};
        _indices.push_back(new_index);

        if (_indices.size() > load_factor * _values->size())
            _resize();

        return _indices.size() - 1;
    }

    
    int StableIndexMap::find_or_insert(const std::vector<int>& value) {
        assert(value.size() <= 2 && !value.empty());

        auto index = _hasher.hashVector(value) % _values->size();
        Entry entry = {value[0], value.size() < 2 ? -1 : value[1]};
        std::vector looking_for = {entry, {}}; // We are looking for the entry or an empty entry

        auto it = circular_find_first_of<Entry>(_values->begin(), _values->end(),
            looking_for.begin(), looking_for.end(), index);
        if (it == _values->end())
            return _values->size();

        auto new_index = std::distance(_values->begin(), it);

        // Entry is not empty
        if (it->left >= 0) {
            return _find_stable_index(new_index);
        }




        // Entry is empty, insert
        _values->operator[](new_index) = entry;
        _indices.push_back(new_index);

        if (_indices.size() > load_factor * _values->size()) {
            _resize();
        }

        return _indices.size() - 1;
    }

    
    bool StableIndexMap::contains(const std::vector<int>& vec) {
        assert(vec.size() <= 2 && !vec.empty());

        std::vector<Entry> entry = {{vec[0], vec.size() < 2 ? -1 : vec[1]}, {}};

        auto index = _hasher.hashVector(vec) % _values->size();
        auto it = circular_find_first_of<Entry>(_values->begin(), _values->end(), entry.begin(), entry.end(), index);

        auto exists = it->left >= 0 and it != _values->end();
        return exists;
    }


    
    int StableIndexMap::find(const std::vector<int>& vec) {
        assert(vec.size() <= 2 && !vec.empty());

        Entry entry = {vec[0], vec.size() < 2 ? -1 : vec[1]};

        std::vector looking_for {{entry, {}}}; // We are looking for the entry or an empty entry

        auto index = _hasher.hashVector(vec) % _values->size();
        auto it = circular_find_first_of<Entry>(_values->begin(), _values->end(),
            looking_for.begin(), looking_for.end(), index);

        if (it == _values->end() || it->left == -1) {
            return -1;
        }
        #ifdef _DEBUG

            auto exists = it->left >= 0 and it != _values->end();
            if (exists) {
                probes += it < (_values->begin() + index) ? std::distance(_values->begin(), it) + std::distance(it, _values->end())
                        : std::distance(_values->begin() + index, it);
                calls++;
            }
            misses += exists ? 0 : 1;

        #endif
        auto raw_index = std::distance(_values->begin(), it);

        return _find_stable_index(raw_index);
    }


    int StableIndexMap::_find_stable_index(size_t raw_index) {
        auto stable_index = std::find(std::execution::unseq,  _indices.begin(), _indices.end(), raw_index);

        return std::distance(_indices.begin(), stable_index);
    }

    
    size_t StableIndexMap::size() {
        return _indices.size();
    }

    void StableIndexMap::print_info() {
        g_log << "Number of stable index map entries: " << size()
            << "\nNumber of entries: " << _values->size()
            << "\nLoad factor: " << load_factor
            << "\nMax size: " << max_size
            << "\nNumber of probes: " << probes
            << "\nNumber of calls: " << calls
            << "\nNumber of misses: " << misses
            << "\nAverage number of probes: " << (float)probes/(float)calls << std::endl;
    }
    
    void StableIndexMap::_resize() {
        g_log << "Resizing StableIndexMap to " << std::min(_values->size() * 2, max_size) << std::endl;

        _resize(std::min(_values->size() * 2, max_size));
    }

    void StableIndexMap::_resize(size_t new_size) {
        assert (new_size > _indices.size());


        auto old_values = std::move(_values);
        _values = std::make_unique<std::vector<Entry>>(new_size);
        for (auto i = 0; i < _indices.size(); i++) {
            auto value = old_values->operator[](_indices[i]);
            auto new_index = _hasher.hashVector({value.left, value.right}) % new_size;
            auto it = circular_find<Entry>(_values->begin(), _values->end(), {}, new_index);
            new_index = std::distance(_values->begin(), it);
            _values->operator[](new_index) = value;
            _indices[i] = new_index;
        }

    }


}
