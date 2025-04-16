#include "stable_index_hash_map.h"

#include "logging.h"

using namespace std;

namespace utils {
int StableIndexMap::find_or_insert(const vector<int> &value) {
    Entry entry = {value[0], value.size() < 2 ? -1 : value[1]};
    auto it = _values.find(entry);
    if (it != _values.end()) {
        return it->second;
    }
    _values.insert({entry, _size});
    return _size++;
}

int StableIndexMap::find(const vector<int> &vec) {
    assert(vec.size() <= 2 && !vec.empty());

    Entry entry = {vec[0], vec.size() < 2 ? -1 : vec[1]};

    auto it = _values.find(entry);
    if (it == _values.end())
        return -1;
    return it->second;
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
