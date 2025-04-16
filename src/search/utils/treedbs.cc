#include "treedbs.h"

using namespace std;

namespace utils {
TreeDBS::TreeDBS(size_t size) : _size(size) {
    assert(size > 0 && "TreeDBS size must be greater than 0");
}

void TreeDBS::insert(const vector<int> &vec) {
    assert(vec.size() == _size && "Vector size must match TreeDBS size");
    put(vec.begin(), vec.end());
}

bool TreeDBS::contains(const vector<int> &vec) {
    assert(vec.size() == _size && "Vector size must match TreeDBS size");
    return find(vec.begin(), vec.end()) >= 0;
}

size_t TreeDBS::size() {
    return entries.size();
}

int TreeDBS::find(const vector<int>::const_iterator begin, const vector<int>::const_iterator end) {
    auto count = distance(begin, end);
    if (count <= 2)
        return entries.find({begin[0], count == 1 ? -1 : begin[1]});

    auto mid = begin + split_range(0, count);
    assert(mid > begin && mid < end && "Invalid midpoint calculation");

    auto left = find(begin, mid);
    if (left == -1)
        return -1;

    auto right = find(mid, end);
    if (right == -1)
        return -1;

    return entries.find({left, right});
}

int TreeDBS::put(const vector<int>::const_iterator begin, const vector<int>::const_iterator end) {
    auto count = distance(begin, end);
    if (count <= 2)
        return entries.find_or_insert({begin[0], count == 1 ? -1 : begin[1]});

    auto mid = begin + split_range(0, count);
    assert(mid > begin && mid < end && "Invalid midpoint calculation");

    auto left = put(begin, mid);
    if (left == -1)
        return -1;

    auto right = put(mid, end);
    if (right == -1)
        return -1;

    return entries.find_or_insert({left, right});
}

void TreeDBS::print_info() {
    entries.print_info();
}
}
