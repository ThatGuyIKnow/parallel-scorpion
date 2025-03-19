#include "loes.h"
#include "cloes.h"
#include <iostream>
#include <cassert>
#include <algorithm>


namespace loes{
Loes::Loes(const BitState &state, bool reuse_tree_levels) : GenericLoes(reuse_tree_levels) {
    data.state_bit_length = state.size();
    data.tree_levels = make_shared<vector<BitSeq>>(data.state_bit_length);
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

Loes::Loes(const BitState &state1, const BitState &state2, bool will_reuse_tree_levels) : GenericLoes(will_reuse_tree_levels)
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

template<class BitStateT>
void GenericLoes::add_state_with_type(const BitStateT &state) {
    size_t depth = 0;
    if (data.is_empty) {
        prev_state = make_unique<SequentialBitState>(data.state_bit_length);
        depth = -1;
        data.is_empty = false;
    }
    else {
        assert(state.size() == data.state_bit_length);
        size_t first_diff = prev_state->first_differing_bit(state);
        if(first_diff == state.size())
            return;
        depth = first_diff;
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

void GenericLoes::update_prev_state(const SequentialBitState &state)
{
    if(SequentialBitState* p = dynamic_cast<SequentialBitState*>(prev_state.get()))
        *p = state;
    else
        prev_state = state.clone();
}

void GenericLoes::update_prev_state(const BitState &state)
{
    prev_state = state.clone();
}

void GenericLoes::add_state(const SequentialBitState &state)
{
    add_state_with_type(state);
    update_prev_state(state);
}

void GenericLoes::add_state(const BitState &state)
{
    add_state_with_type(state);
    update_prev_state(state);
}

size_t GenericLoes::path_offset(const BitState &state) const {
    if(!data.is_compiled)
        compile();
    size_t offset = 0;
    size_t s_size = state.size();
    size_t d_size = data.size();
    for (size_t depth = 0; depth < s_size; ++depth) {
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


size_t GenericLoes::member_index(const BitState &state) const
{
    size_t offset = path_offset(state);
    if (offset == (size_t) -1)
        return -1;
    return data.rank(offset) - data.rank(data.level_offsets[data.state_bit_length - 1] - 1) - 1;
}

bool GenericLoes::contains(const BitState &state) const {
    return path_offset(state) != (size_t) -1;
}

size_t GenericLoes::size() const
{
    if(!data.is_compiled)
        compile();
    return data.rank(data.level_offsets[data.state_bit_length] - 1) - data.rank(data.level_offsets[data.state_bit_length - 1] - 1);
}

size_t GenericLoes::bit_size() const
{
    return data.size();
}

void GenericLoes::compile() const
{
    assert(!data.is_compiled);
    if(!data.level_offsets_computed)
        data.compute_level_offsets();
    data.compile_combined_treelevels();
    prev_state.reset();
    data.is_compiled = true;
}

void GenericLoes::print(bool full) const{
    cout << "-----" << endl;
    cout << "Static LOES with "<< size() << " states of length " << data.state_bit_length << ", represented in " << data.size() << " bits." << endl;
    if (full) {        
        vector<string> s(data.state_bit_length);
        s.back() = "";
        size_t last_level = data.state_bit_length - 1;
        for (size_t i = 0; i < data.level_size(last_level); ++i)
        {
            if (i % 2 == 0 && i != 0)
                s.back().push_back(' ');
            if (data.get(last_level, i))
                s.back().push_back('1');
            else 
                s.back().push_back('0');
        }
        for (size_t level = data.state_bit_length - 1; level > 0; level--)
        {
            s[level - 1] = string(s.back().size(), ' ');
            size_t i = 0;
            for (size_t j = 0; j < s[level].size() && i < data.level_size(level - 1) - 1;)
            {
                while (j < s[level].size() && s[level][j] == ' ') {
                    j++;
                }
                int bit1i = j;
                bool bit1 = data.get(level - 1, i);
                i++;
                j++;
                while (j < s[level].size() && s[level][j] == ' ') {
                    j++;
                }
                int bit2i = j;
                j++;
                bool bit2 = data.get(level - 1, i);
                i++;
                if (bit1 && !bit2)
                {
                    s[level - 1][bit1i] = '1';
                    s[level - 1][bit2i] = '0';

                }
                else if (!bit1 && bit2) {
                    s[level - 1][bit1i] = '0';
                    s[level - 1][bit2i] = '1';
                }
                else {
                    while (j < s[level].size() && s[level][j] == ' ') {
                        j++;
                    }
                    int bit3i = j;
                    j++;
                    while (j < s[level].size() && s[level][j] == ' ') {
                        j++;
                    }
                    j++;
                    s[level - 1][bit2i] = '1';
                    s[level - 1][bit3i] = '1';

                }
            }
        }

        for (auto &str : s) {
            int cnt = 0;
            for (auto &ch : str)
            {
                if (ch == '1' || ch == '0')
                {
                    cnt++;
                }
                else if (cnt != 0 && cnt % 2)
                {
                    ch = '-';//(char) 196;
                }

            }
        }
        for (auto &str : s)
        {
            cout << str << endl;
        }
    }
    cout << "-----" << endl ;

}
template<class LoesT>
void GenericLoes::merge(const LoesT &loes1, const LoesT &loes2, LoesT &new_loes)
{
    assert(loes1.data.level_offsets_computed);
    assert(loes2.data.level_offsets_computed);
    assert(loes1.data.state_bit_length == loes2.data.state_bit_length);
    data.state_bit_length = loes1.data.state_bit_length;
    if(loes1.can_reuse_tree_levels())
        new_loes.data.tree_levels = loes1.data.tree_levels;
    else if(loes2.can_reuse_tree_levels())
        new_loes.data.tree_levels = loes2.data.tree_levels;
    else
        new_loes.data.tree_levels = make_shared<vector<BitSeq>>(data.state_bit_length);
    auto it1 = loes1.begin();
    auto it2 = loes2.begin();
    
    auto end1 = loes1.end();
    auto end2 = loes2.end();

    loes::SequentialBitState state1(loes1.data.state_bit_length);
    loes::SequentialBitState state2(loes2.data.state_bit_length);

    bool done = false;
    
    while (!done)
    {        
        if (it1 == end1)
        {
            for (; it2 != end2; ++it2){
                it2.deref_into(state2);
                new_loes.add_state(state2);
                done = true;
            }
                 
        }
        else if (it2 == end2)
        {
            for (; it1 != end1; ++it1){
                it1.deref_into(state1);
                new_loes.add_state(state1);
                done = true;
            }
                
        } else {
            it1.deref_into(state1);
            it2.deref_into(state2);
            if (state1 < state2)
            {
                new_loes.add_state(state1);
                ++it1;
            }
            else
            {
                new_loes.add_state(state2);
                ++it2;
            }
        }
    }
    new_loes.data.compute_level_offsets();
}
template void GenericLoes::merge<Loes>(const Loes &loes1, const Loes &loes2, Loes &new_loes);
template void GenericLoes::merge<Cloes>(const Cloes &loes1, const Cloes &loes2, Cloes &new_loes);

Loes::iterator Loes::begin() const {
    vector<size_t> offset_state;
    if(data.has_combined_tree_levels){
        offset_state = data.level_offsets;
        offset_state.pop_back();    
        
        for (auto &offset : offset_state)
        {
            while (!data[offset])
            {
                ++offset;
                if (offset>= data.size())
                    return iterator(this);
            }
        }
    } else {
        offset_state = vector<size_t>(data.state_bit_length, 0);    
        size_t level = 0;
        for (auto &offset : offset_state)
        {
            while (!data.get(level, offset)) {
                ++offset;
                if (offset>= data.level_size(level))
                    return iterator(this);
            }
            ++level;
        }
    }
    return iterator(this, offset_state, data.has_combined_tree_levels);
}

Loes::iterator Loes::end() const {
    return iterator(this);
}




void Loes::iterator::deref_into(SequentialBitState &state) const {
    assert(!is_end);
    static const size_t one = size_t(1);
    for (size_t i = min_updated_level; i < offset_state.size(); ++i)
    {
        state.set(i, offset_state[i] & one);
    }
}

size_t Loes::iterator::incr_seperate_levels() {
    assert(!is_end);
    size_t level = loes->data.state_bit_length - 1;
    for (; level != (size_t) -1; level--) {
        size_t &offset = offset_state[level];
        size_t rec = offset / 2;
        do {
            ++offset;
            if (offset >= loes->data.level_size(level)) {
                is_end = true;
                return level;
            }
        } while (!loes->data.get(level, offset));
        if (rec == (offset / 2))
            break;
    }
    return level;
}

size_t Loes::iterator::incr_combined_levels() { 
    assert(!is_end);
    size_t level = loes->data.state_bit_length - 1;
    for (; level != (size_t) -1; level--) {
        size_t &offset = offset_state[level];
        size_t rec = offset / 2;
        do {
            ++offset;
            if (offset >= loes->data.size()) {
                is_end = true;
                return level;
            }
        } while (!loes->data[offset]);
        if (rec == (offset / 2))
            break;
    }
    return level;
}


Loes::iterator& Loes::iterator::operator++() {
    if(combined_tree_levels)
        min_updated_level = incr_combined_levels();
    else
        min_updated_level = incr_seperate_levels();
    return *this;
}
Loes::iterator Loes::iterator::operator++(int) {
    Loes::iterator tmp = *this;
    ++(*this);
    return tmp;
}

bool operator==(const Loes::iterator& a, const Loes::iterator& b)
{
    return(a.loes == b.loes) && ((a.is_end && b.is_end) || (a.offset_state == b.offset_state));
}

bool operator!=(const Loes::iterator& a, const Loes::iterator& b) {
    return !(a == b);
}

void Loes::Data::subblock_popcount_push_back(size_t popcount)
{
    static const size_t block_size = 1 << LOG2_BLOCK_SIZE;
    static const size_t subblock_size = 1 << LOG2_SUBBLOCK_SIZE;

    if (subblock_popcount.size() * subblock_size >= block_popcount.size() * block_size)
    {
        block_popcount.push_back(block_popcount.back() + subblock_popcount.back() + popcount);
        subblock_popcount.push_back(0);
    }
    else {
        subblock_popcount.push_back(subblock_popcount.back() + popcount);
    }
}

bool Loes::Data::operator[](size_t idx) const
{
    if(has_combined_tree_levels){
        return combined_tree_levels[idx]; 
    }
    size_t level = getLevelWithOffset(idx);
    if (level == 0)
    {
        return (*tree_levels)[level][idx];
    }
    return (*tree_levels)[level][idx - level_offsets[level]];
}

size_t Loes::Data::rank(size_t offset) const
{
    assert(offset < size());
    size_t count = block_popcount[offset >> LOG2_BLOCK_SIZE]; // offset >> LOG2_BLOCK_SIZE = floor(offset/BLOCK_SIZE)
    count += subblock_popcount[offset >> LOG2_SUBBLOCK_SIZE]; // offset >> LOG2_SUBBLOCK_SIZE = floor(offset/SUBBLOCK_SIZE)
    
    size_t end_offset = offset;
    static const size_t bit_mask = ((1 << LOG2_SUBBLOCK_SIZE) - 1);
    size_t start_offset = end_offset - (offset & bit_mask); //end_offset - (offset mod SUBBLOCK_SIZE)

    return count + combined_tree_levels.popcount(end_offset + 1, start_offset);
}

size_t Loes::Data::size() const
{
    return level_offsets.back();
}

size_t Loes::Data::getLevelWithOffset(size_t offset) const
{
    size_t high = level_offsets.size();
    assert(offset < level_offsets[high - 1]);
    size_t low = 0;
    size_t mid = (high + low) / 2;

    while (low < high - 1) //Binary search
    {
        if (offset < level_offsets[mid - 1])
            high = mid;
        else
            low = mid;
        mid = (high + low) / 2;
    }
    return low - 1;
}

size_t Loes::Data::level_size(size_t level) const
{
    return level_offsets[level + 1] - level_offsets[level];
}

size_t Loes::Data::get(size_t level, size_t idx) const
{
    assert(level < state_bit_length);
    if (has_combined_tree_levels)
        return combined_tree_levels[level_offsets[level] + idx];
    return (*tree_levels)[level][idx];
}

void Loes::Data::compile_combined_treelevels()
{
    for (size_t i = 0; i < state_bit_length; i++)
    {
        BitSeq &level = (*tree_levels)[i];
        combined_tree_levels.push_back(level);
        if(reuse_tree_levels)
            level.resize(0);
        else
            level.reset();
    }
    unsigned subblock_size = 1 << LOG2_SUBBLOCK_SIZE;
    size_t i = 0;
    for (; i + subblock_size < combined_tree_levels.size(); i += subblock_size)
        subblock_popcount_push_back(combined_tree_levels.popcount(i + subblock_size, i));
    subblock_popcount_push_back(combined_tree_levels.popcount(combined_tree_levels.size(), i));
    if(!reuse_tree_levels)
        tree_levels.reset();
    has_combined_tree_levels = true;
}

void GenericLoes::Data::compute_level_offsets()
{
    assert(!is_compiled);
    size_t level_offset = 0;
    for (size_t i = 0; i < state_bit_length; i++)
    {
        BitSeq &level = (*tree_levels)[i];
        level_offset += level.size();
        level_offsets.push_back(level_offset);
    }
    
    level_offsets_computed = true;
}
}