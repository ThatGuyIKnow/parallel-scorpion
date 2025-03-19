#include "merge_container.h"

namespace merge_container {

template<class Container>
MergeContainerStructure<Container>::MergeContainerStructure(const BitState &state) {
    add_state(state);
}


template<class Container>
void MergeContainerStructure<Container>::add_state(const BitState &state) 
{
    if(has_state) {
        if(conts.begin()->merge_count == 1 && !conts.empty()) {
            Container cont(state, *stored_state);
            merge(cont);
        } else {
            conts.emplace_front(state, *stored_state, 1, reuse_tree_levels);
        }
        has_state = false;
    } else {
        if(conts.empty())
            state_bit_length = state.size();
        stored_state = state.clone();
        has_state = true;
    }
    assert(state.size() == state_bit_length);
}

template<class Container>
bool MergeContainerStructure<Container>::contains(const BitState &state) const
{
    for (auto i = conts.rbegin(); i != conts.rend(); ++i)
    {
        if (i->cont.contains(state))
        {
            return true;
        }
    }
    if(has_state && state == *stored_state)
        return true;
    return false;
}

template<class Container>
void MergeContainerStructure<Container>::print(bool full) const
{
    size_t num_states = has_state;
    size_t num_bits = has_state * state_bit_length;
    for (auto i = conts.begin(); i != conts.end(); ++i)
    {
        num_states += i->cont.size();
        num_bits += i->cont.bit_size();
    }
    cout << num_states << " states of length " << state_bit_length << ", represented in " << num_bits << " bits." << endl;
    
    if(full){
        for (auto i = conts.begin(); i != conts.end(); ++i)
        {
            cout << endl << "merge count: " << i->merge_count << endl;
            i->cont.print(full);
        }
    } 
}

template<class Container>
void MergeContainerStructure<Container>::merge(Container& cont)
{
    auto merged_cont = conts.begin();
    if(conts.size() == 1){
        conts.emplace(next(merged_cont), merged_cont->merge_count + 1, cont, merged_cont->cont, false);
        conts.pop_front();
    } else if (next(merged_cont)->merge_count == merged_cont->merge_count + 1) {
        Container cont_new = {cont, merged_cont->cont};
        conts.pop_front();
        merge(cont_new);
    } else {
        conts.emplace(next(merged_cont), merged_cont->merge_count + 1, cont, merged_cont->cont, reuse_tree_levels);
        conts.pop_front();
    }
}
template class MergeContainerStructure<loes::Loes>;
template class MergeContainerStructure<loes::Cloes>;
}