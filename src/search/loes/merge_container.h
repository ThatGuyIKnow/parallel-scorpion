#ifndef LOES_DYNAMIC_LOES_H
#define LOES_DYNAMIC_LOES_H

#include "cloes.h"

namespace merge_container{
using BitState = loes::BitState;
template<class Container> class MergeContainerStructure
{
    struct MergedContainer {
        Container cont;
        size_t merge_count = 0;
        MergedContainer(size_t merge_count, const Container &cont1, const Container &cont2, bool reuse_tree_levels) : cont(cont1, cont2, reuse_tree_levels), merge_count(merge_count) {}
        MergedContainer(const BitState &state1, const BitState &state2, size_t merge_count, bool reuse_tree_levels) : cont(state1, state2, reuse_tree_levels), merge_count(merge_count) {}
    };
public:
    MergeContainerStructure(bool reuse_tree_levels) : reuse_tree_levels(reuse_tree_levels){}
    MergeContainerStructure(const BitState &state);
    void add_state(const BitState &state);
    bool contains(const BitState &state) const;
    void print(bool full = false) const;
private:
    size_t state_bit_length = 0;
    list<MergedContainer> conts = {};
    unique_ptr<BitState> stored_state;
    bool has_state = false;
    bool reuse_tree_levels;
    void merge(Container& cont);
};
}
#endif