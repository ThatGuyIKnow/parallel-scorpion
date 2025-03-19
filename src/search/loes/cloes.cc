#include "cloes.h"
#include <cassert>
#include <algorithm>
#include <iostream>

using namespace std;
namespace loes{
size_t Cloes::path_offset(const BitState &state) const
{
    if(!data.is_compiled)
        compile();
    size_t offset = 0;
    size_t d_size = data.size();
    size_t s_size = state.size();
    for (size_t depth = 0; depth < s_size; ++depth) {
        if (offset + 1 < d_size && !data[offset] && !data[offset + 1]) {
            return offset;
        }
        offset += state[depth];
        if (offset >= d_size || !data[offset]) {
            break;
        }
        if (depth == data.state_bit_length - 1) {
            return offset;
        }
        offset = data.rank(offset) << 1; // rank(offset) * 2
    }
    return -1;
}

template<class BitStateT>
void Cloes::add_state_with_type(const BitStateT &state) {
    ++m_size;
    size_t depth = 0;
    if (data.is_empty) {
        prev_state = make_unique<SequentialBitState>(data.state_bit_length);
        depth = -1;
        data.is_empty = false;
    }
    else {
        assert(state.size() == data.state_bit_length);
        size_t first_diff = prev_state->first_differing_bit(state);
        if(first_diff == state.size()) {
            m_size--;
            return;
        }
        depth = first_diff;
        if (depth == data.state_bit_length - 1)
        {      
            BitSeq &depth_level = (*data.tree_levels)[depth];      
            size_t len = depth_level.size();
            depth_level.set(len - 1, 0);
            depth_level.set(len - 2, 0);
            for (int i = depth - 1; i >= 0; i--)
            {
                BitSeq &curr_level = (*data.tree_levels)[i];
                BitSeq &below_level = (*data.tree_levels)[i + 1];
                size_t len1 = curr_level.size();
                size_t len2 = below_level.size();
                
                if ((len2 < 4) || !(curr_level[len1 - 1] && curr_level[len1 - 2] && 
                (!below_level[len2 - 1] && !below_level[len2 - 2] &&
                 !below_level[len2 - 3] && !below_level[len2 - 4]) ))
                {
                    break;
                }
                curr_level.set(len1 - 1, 0);
                curr_level.set(len1 - 2, 0);

                below_level.pop_back();
                below_level.pop_back();
                below_level.pop_back();
                below_level.pop_back();
            }            
            return;
        }
        
        (*data.tree_levels)[depth].set_back(1);
    }

    for (size_t i = depth + 1; i < data.state_bit_length; ++i) {
        BitSeq &level = (*data.tree_levels)[i];
        if (state[i]) {
            level.push_back(0);
            level.push_back(1);
        } else {
            level.push_back(1);
            level.push_back(0);
        }
    }
}


void Cloes::add_state(const SequentialBitState &state)
{
    add_state_with_type(state);
    update_prev_state(state);
}


void Cloes::add_state(const BitState &state)
{
    add_state_with_type(state);
    update_prev_state(state);
}

Cloes::Cloes(const BitState &state, bool reuse_tree_levels) : GenericLoes(reuse_tree_levels) {
    data.state_bit_length = state.size();
    data.tree_levels = make_shared<vector<BitSeq>>(data.state_bit_length);
    m_size = 1;
    for (size_t i = 0; i < data.state_bit_length; ++i) {
        BitSeq &level = (*data.tree_levels)[i];
        if (state[i]) {
            level.push_back(0);
            level.push_back(1);
        } else {
            level.push_back(1);
            level.push_back(0);
        }
    }
    data.is_empty = false;
    data.compute_level_offsets();
}

Cloes::Cloes(const BitState &state1, const BitState &state2, bool will_reuse_tree_levels) : GenericLoes(will_reuse_tree_levels)
{
    assert(state1.size() == state2.size());
    data.state_bit_length = state1.size();
    data.tree_levels = make_shared<vector<BitSeq>>(data.state_bit_length);
    if(state1 < state2){
        add_state(state1);
        add_state(state2);
    } else {
        add_state(state2);
        add_state(state1);
    }
    data.compute_level_offsets();
}

//TODO: Fix this print, Loes looks way nicer
void Cloes::print(bool full) const
{
    cout << "-----" << endl;
    cout << "Static cLOES with " << size() << " states of length " << data.state_bit_length << ", represented in " << data.size() << " bits." << endl;
    if (full) {
        size_t max_size = 0;
        for (size_t i = 0; i < data.state_bit_length; ++i)
        {
            max_size = max(data.level_size(i), max_size);
        }
        for (size_t i = 0; i < data.state_bit_length; ++i)
        {
            assert(max_size >= data.level_size(i));
            size_t num_space = (max_size- data.level_size(i)) / 2;

            for (size_t j = 0; j < num_space; j++)
            {
                cout << " ";
            }
            
            for (size_t j = 0; j < data.level_size(i); ++j)
            {
                cout << data.get(i,j);
            }
            cout << endl;
        }
    }
    cout << "-----" << endl ;
}
size_t Cloes::size() const
{
    return m_size;
}

Cloes::iterator Cloes::begin() const {
    bool found_comp = false;
    vector<size_t> offset_state;
    size_t comp_level = data.state_bit_length;

    if(data.has_combined_tree_levels) {
        offset_state = data.level_offsets;
        offset_state.pop_back();

        for (size_t level = data.state_bit_length - 1; level != (size_t) -1; level--) {
            if(!data.level_size(level))
                continue;
            size_t &offset = offset_state[level];
        
            while (!data[offset]) {
                if(!(offset % 2) && !data[offset] && !data[offset + 1]) {
                    found_comp = true;
                    comp_level = level;
                    break;
                }
                ++offset;
                if (offset>= data.size())
                    return iterator(this);
            }
        }
    } else {
        offset_state = vector<size_t>(data.state_bit_length, 0);
        for (size_t level = data.state_bit_length - 1; level != (size_t) -1; level--) {
            if(!data.level_size(level))
                continue;
            size_t &offset = offset_state[level];
        
            while (!data.get(level, offset)) {
                if(!(offset % 2) && !data.get(level, offset) && !data.get(level, offset + 1)) {
                    found_comp = true;
                    comp_level = level;
                    break;
                }
                ++offset;
                if (offset >= data.level_size(level))
                    return iterator(this);
            }
        }
    }
    if(found_comp)
        return iterator(this, offset_state, comp_level, data.has_combined_tree_levels);
    
    return iterator(this, offset_state, data.has_combined_tree_levels);
}

Cloes::iterator Cloes::end() const {
    return iterator(this);
}

size_t Cloes::iterator::incr_seperate_levels() {
    assert(!is_end);
    size_t start_level = cloes->data.state_bit_length - 1;
    if (found_comp) {
        ++comp_count;
        ++comp_count_state;
        if (comp_count < end_count)
            return comp_level;
        found_comp = false;
        start_level = comp_level;
        
        for (size_t level = comp_level + 1; level < cloes->data.state_bit_length; ++level) {
            size_t &offset = offset_state[level];
            if((offset < cloes->data.level_size(level)) && !(offset % 2) && !cloes->data.get(level, offset) && !cloes->data.get(level, offset + 1)) {
                report_compression(level);
                break;
            }
        }
        if(!found_comp)
            comp_level = cloes->data.state_bit_length;
    }
    size_t level = start_level;
    for (; level != (size_t) -1; level--) {
        size_t &offset = offset_state[level];
        size_t rec = offset / 2;
        do {
            ++offset;
            if (offset >= cloes->data.level_size(level)) {
                if(level == 0) {
                    is_end = true;
                    return level;
                }
                break;
            }
            if(!(offset % 2) && !cloes->data.get(level, offset) && !cloes->data.get(level, offset + 1)) {
                report_compression(level);
                break;
            }
        } while (!cloes->data.get(level, offset));
        if (rec == (offset / 2))
            break;
            
    }
    return level;
}

size_t Cloes::iterator::incr_combined_levels() {
    assert(!is_end);
    size_t start_level = cloes->data.state_bit_length - 1;
    if (found_comp) {
        ++comp_count;
        ++comp_count_state;
        if (comp_count < end_count)
            return comp_level;
        found_comp = false;
        start_level = comp_level;
        
        for (size_t level = comp_level + 1; level < cloes->data.state_bit_length; ++level) {
            size_t &offset = offset_state[level];
            if((offset < cloes->data.level_offsets[level + 1]) && !(offset % 2) && !cloes->data[offset] && !cloes->data[offset + 1]) {
                report_compression(level);
                break;
            }
        }
        if(!found_comp)
            comp_level = cloes->data.state_bit_length;
    }
    size_t level = start_level;
    for (; level != (size_t) -1; level--) {
        size_t &offset = offset_state[level];
        size_t rec = offset / 2;
        do {
            ++offset;
            if (offset >= cloes->data.level_offsets[level + 1]) {
                if(level == 0) {
                    is_end = true;
                    return level;
                }
                break;
            }
            if(!(offset % 2) && !cloes->data[offset] && !cloes->data[offset + 1]) {
                report_compression(level);
                break;
            }
        } while (!cloes->data[offset]);
        if (rec == (offset / 2))
            break;
    }
    return level;
}

Cloes::iterator& Cloes::iterator::operator++() {
    if(combined_tree_levels)
        min_updated_level = incr_combined_levels();
    else
        min_updated_level = incr_seperate_levels();
    return *this;
}
Loes::iterator Cloes::iterator::operator++(int){
    iterator tmp = *this;
    ++(*this);
    return tmp;
}

void Cloes::iterator::deref_into(SequentialBitState &state) const { 
    assert(!is_end);
    static const size_t one = size_t(1);
    for (size_t i = comp_level; i < cloes->data.state_bit_length; i++)
            state.set(i, comp_count_state[i]);
    for (size_t i = min_updated_level; i < comp_level; ++i)
            state.set(i, offset_state[i] & one);
}
void Cloes::iterator::report_compression(size_t level)
{
    found_comp = true;
    comp_level = level;
    comp_count = 0;
    end_count = (size_t) (1 << (cloes->data.state_bit_length - comp_level));

    comp_count_state = SequentialBitState(cloes->data.state_bit_length);
}
}