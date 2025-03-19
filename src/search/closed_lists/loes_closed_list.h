#ifndef CLOSED_LISTS_LOES_CLOSED_LIST_H
#define CLOSED_LISTS_LOES_CLOSED_LIST_H

#include "../task_proxy.h"
#include "../closed_list.h"
#include "../loes/cloes.h"
#include "../loes/merge_container.h"
#include <vector>

using namespace std;

namespace loes_closed_list {
template<class LoesType = loes::Loes>
class LoesClosedList : public ClosedList {
    using Loes = merge_container::MergeContainerStructure<LoesType>;
    TaskProxy task_proxy;
    Loes c_list;
    const int_packer::IntPacker &state_packer;
    vector<unsigned int> var_bit_lengths = {};
    shared_ptr<loes::BitMap> bitmap;
    int state_bit_length;
    bool full_print;

    loes::MappedBitState state_to_bitstate(const State &state) const;
    vector<size_t> min_entropy_bitorder(size_t num_samples, size_t max_sample_iterations) const;

public:
    explicit LoesClosedList(const options::Options &opts);

    void add_state(const State &state);
    bool contains_state(const State &state) const;
    void print() const;
};

extern void add_loes_closed_list_options_to_parser(
    options::OptionParser &parser);
}

#endif
