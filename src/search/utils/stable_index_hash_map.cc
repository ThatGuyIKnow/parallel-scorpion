#include "stable_index_hash_map.h"

#include "logging.h"

using namespace std;

namespace utils {
int StableIndexMap::insert(const vector<int> &value) {
    assert(value.size() <= 2 && !value.empty());
    Entry entry = {value[0], value.size() < 2 ? -1 : value[1]};

    _values.insert({entry, _size});
    return _size++;
}

int StableIndexMap::find_or_insert(const vector<int> &value) {
    Entry entry = {value[0], value.size() < 2 ? -1 : value[1]};
    auto it = _values.find(entry);
    if (it != _values.end()) {
        return it->second;
    }
    _values.insert({entry, _size});
    return _size++;
}

bool StableIndexMap::contains(const vector<int> &vec) {
    assert(vec.size() <= 2 && !vec.empty());

    Entry entry = {vec[0], vec.size() < 2 ? -1 : vec[1]};
    return _values.contains(entry);
}

int StableIndexMap::find(const vector<int> &vec) {
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
          << "\nNumber of probes: " << probes
          << "\nNumber of calls: " << calls
          << "\nNumber of misses: " << endl;
}
}
