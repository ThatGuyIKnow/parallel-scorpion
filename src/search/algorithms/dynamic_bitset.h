#ifndef ALGORITHMS_DYNAMIC_BITSET_H
#define ALGORITHMS_DYNAMIC_BITSET_H

#include <cassert>
#include <limits>
#include <vector>
#include <bits/stdc++.h>
#include "libpopcnt.h"

/*
  Poor man's version of boost::dynamic_bitset, mostly copied from there.
*/
using namespace std;
//TODO: Thoroughly all newly implemented functions
namespace dynamic_bitset {
template<typename Block = unsigned int>
class DynamicBitset {
    static_assert(
        !numeric_limits<Block>::is_signed,
        "Block type must be unsigned");

    vector<Block> blocks;
    size_t num_bits;

    static const Block zeros;
    static const Block ones;

    static const int bits_per_block = numeric_limits<Block>::digits;

    static int compute_num_blocks(size_t num_bits) {
        return num_bits / bits_per_block +
               static_cast<int>(num_bits % bits_per_block != 0);
    }

    static size_t block_index(size_t pos) {
        return pos / bits_per_block;
    }

    static size_t bit_index(size_t pos) {
        return pos % bits_per_block;
    }

    static Block bit_mask(size_t pos) {
        return Block(1) << bit_index(pos);
    }

    int count_bits_in_last_block() const {
        return bit_index(num_bits);
    }

    void zero_unused_bits() {
        const int bits_in_last_block = count_bits_in_last_block();

        if (bits_in_last_block != 0) {
            assert(!blocks.empty());
            blocks.back() &= ~(ones << bits_in_last_block);
        }
    }
    template <typename T>
	size_t popcountWithType(size_t& i, size_t to) const;

    template <typename OtherBlock>
    static size_t block_index(size_t pos) {
        static const int bits_per_other_block = numeric_limits<OtherBlock>::digits;
        return pos / bits_per_other_block;
    }
    static Block bit_mask(int from, int to) {
        // Return mask with all bits in the range [from, to) set to 1.
        assert(from >= 0 && to >= from && to <= bits_per_block);
        int length = to - from;
        if (length == bits_per_block) {
            // 1U << BITS_PER_BIN has undefined behaviour in C++; e.g.
            // 1U << 32 == 1 (not 0) on 32-bit Intel platforms. Hence this
            // special case.
            assert(from == 0 && to == bits_per_block);
            return ~Block(0);
        } else {
            return ((Block(1) << length) - 1) << from;
        }
    }

public:
    explicit DynamicBitset(size_t num_bits)
        : blocks(compute_num_blocks(num_bits), zeros),
          num_bits(num_bits) {
    }

    template<typename OtherBlock>
    explicit DynamicBitset(const OtherBlock* data, size_t num_bits)
        : blocks((Block*) data, ((Block*) data) + compute_num_blocks(num_bits)),
          num_bits(num_bits) {
        static_assert(
        !numeric_limits<OtherBlock>::is_signed,
        "Block type must be unsigned");
        zero_unused_bits();
    }

    template<typename OtherBlock>
    explicit DynamicBitset(const vector<OtherBlock> &data, size_t num_bits)
        : DynamicBitset(data.data(), num_bits) {
        assert(num_bits <= data.size() * numeric_limits<OtherBlock>::digits);
    }
    
    template<typename OtherBlock>
    explicit DynamicBitset(const DynamicBitset<OtherBlock> &other)
        : DynamicBitset(other.data(), other.size()) {
    }

    explicit DynamicBitset(size_t num_bits, bool val)
        :  num_bits(num_bits) {
        if(val){
            blocks = vector<Block>(compute_num_blocks(num_bits), ones);
            zero_unused_bits();
        } else {
            blocks = vector<Block>(compute_num_blocks(num_bits), zeros);
        }
    }

    explicit DynamicBitset()
        : blocks(),
          num_bits(0) {
    }

    size_t size() const {
        return num_bits;
    }

    /*
      Count the number of set bits.

      The computation could be made faster by using a more sophisticated
      algorithm (see https://en.wikipedia.org/wiki/Hamming_weight).
    */
    int count() const {
        int result = 0;
        for (size_t pos = 0; pos < num_bits; ++pos) {
            result += static_cast<int>(test(pos));
        }
        return result;
    }

    void set() {
        fill(blocks.begin(), blocks.end(), ones);
        zero_unused_bits();
    }

    void reset() {
        fill(blocks.begin(), blocks.end(), zeros);
    }

    void reset(size_t from, size_t to) {
        assert(from <= to);
        assert(to <= num_bits);

        size_t block_from = block_index(from);
        size_t block_to = block_index(to);
        if(block_from == block_to){
            blocks[block_from] &= ~bit_mask(bit_index(from), bit_index(to));
            return;
        }

        blocks[block_from] &= ~bit_mask(bit_index(from), bits_per_block);
        if(block_from + 1 != block_to)
            fill(blocks.begin() + block_from + 1, blocks.begin() + block_to - 1, zeros);
        blocks[block_to] &= ~bit_mask(0, bit_index(to));
    }

    void set(size_t pos, bool val = true) {
        assert(pos < num_bits);
        if(val)
            blocks[block_index(pos)] |= bit_mask(pos);
        else
            reset(pos);
    }

    void reset(size_t pos) {
        assert(pos < num_bits);
        blocks[block_index(pos)] &= ~bit_mask(pos);
    }

    bool test(size_t pos) const {
        assert(pos < num_bits);
        return (blocks[block_index(pos)] & bit_mask(pos)) != 0;
    }

    bool operator[](size_t pos) const {
        return test(pos);
    }

    bool intersects(const DynamicBitset &other) const {
        assert(size() == other.size());
        for (size_t i = 0; i < blocks.size(); ++i) {
            if (blocks[i] & other.blocks[i])
                return true;
        }
        return false;
    }

    bool is_subset_of(const DynamicBitset &other) const {
        assert(size() == other.size());
        for (size_t i = 0; i < blocks.size(); ++i) {
            if (blocks[i] & ~other.blocks[i])
                return false;
        }
        return true;
    }

    size_t first_differing_bit(const DynamicBitset &other) const { 
        size_t min_size = min(blocks.size(), other.blocks.size());
        size_t count = 0;
        for (size_t i = 0; i < min_size; i++)
        {
            if (blocks[i] != other.blocks[i])
            {
                return count + ffs(blocks[i] ^ other.blocks[i]) - 1;
            }
            count += bits_per_block;
        }
        return size();
    }

    size_t first_one_bit() const { 
        size_t count = 0;
        for (size_t i = 0; i < blocks.size(); i++)
        {
            if (blocks[i] != zeros)
            {
                return count + ffs(blocks[i]) - 1;
            }
            count += bits_per_block;
        }
        return size();
    }

    size_t first_zero_bit() const { 
        size_t count = 0;
        for (size_t i = 0; i < blocks.size(); i++)
        {
            if (blocks[i] != ones)
            {
                return count + ffs(~blocks[i]) - 1;
            }
            count += bits_per_block;
        }
        return size();
    }


    size_t last_zero_bit() const { 
        size_t count = blocks.size() * bits_per_block - 1;
        
        auto block = blocks.rbegin();
        size_t last = bit_index(num_bits);
        if(last == 0)
            last = bits_per_block;
        if (*block != bit_mask(0, last))
        {
            count = num_bits - 1;
            for (size_t i = bit_index(num_bits - 1); ((~(*block)) & bit_mask(i)) == 0; i--)
                count--;
            return count;
        }
        block++;
        count -= bits_per_block;
        for (; block != blocks.rend(); block++)
        {
            if (*block != ones)
            {
                for ( size_t i = bits_per_block - 1; ((~(*block)) & bit_mask(i)) == 0; i--)
                    count--;
                return count;
            }
            count -= bits_per_block;
        }
        return size();
    }

    void set_back(bool val) {
        set(size() - 1, val);
    }
    bool back() const {
        return operator[](size() -1);
    }
    void push_back(bool val) {
        if(!count_bits_in_last_block()){
            blocks.push_back(0);
        }
        num_bits++;
        set_back(val);
    }
	bool pop_back() {
        bool val = operator[](size() -1);
        set_back(0);
        num_bits--;
        if(!count_bits_in_last_block()){
            blocks.pop_back();
        }
        return val;
    }
    void push_back(const DynamicBitset<Block> &other);

    void resize(size_t new_size){
        num_bits = new_size;
        blocks.resize(compute_num_blocks(num_bits));
        zero_unused_bits();
    }

    Block* data() {
        return blocks.data();
    }
	const Block* data() const {
        return blocks.data();
    }
    
    bool operator==(const DynamicBitset& other) const {
        return (num_bits == other.num_bits) && (blocks == other.blocks);
    }
    bool operator!=(const DynamicBitset& other) const {
        return !(*this == other);
    }

    /*
    NOTE: These operations counts the first bit (with lowest index) as the most significant bit.
    This differs from boost::dynamic_bitset that counts the last bit as most significant    
    */
    bool operator<(const DynamicBitset<Block>& other) const;
    bool operator>(const DynamicBitset<Block>& other) const{
        return other < *this;
    }
    bool operator>=(const DynamicBitset<Block>& other) const{
        return !(*this < other);
    }

    bool operator<=(const DynamicBitset<Block>& other) const{
        return !(*this > other);
    }
    DynamicBitset<Block>& operator++() {
        size_t zero_bit = last_zero_bit();
        if(zero_bit == size()){
            reset();
            return *this;
        }
        reset(zero_bit, num_bits);
        set(zero_bit);
        return *this;
    }
    DynamicBitset<Block> operator++(int){
        DynamicBitset<Block> tmp = *this;
        ++(*this);
        return tmp;
    }

    size_t popcount() const
    {
        return popcount(num_bits);
    }

    size_t popcount_from(size_t from) const
    {
        return popcount(num_bits, from);
    }

	size_t popcount(size_t to, size_t from = 0) const;

};

template<typename Block>
const Block DynamicBitset<Block>::zeros = Block(0);

template<typename Block>
// MSVC's bitwise negation always returns a signed type.
const Block DynamicBitset<Block>::ones = Block(~Block(0));

template <typename Block>
inline void DynamicBitset<Block>::push_back(const DynamicBitset<Block> &other)
{
    if(!other.size())
        return;
    zero_unused_bits();
    int lshift = count_bits_in_last_block();
    int rshift = bits_per_block - lshift;
    num_bits += other.size();
    blocks.reserve(compute_num_blocks(num_bits));
    if(!lshift){
        blocks.insert(blocks.end(), other.blocks.begin(), other.blocks.end());
        zero_unused_bits();
        return;
    }
    blocks.back() |= (other.blocks.front() << lshift);

    if(other.size() <= (size_t) rshift){
        zero_unused_bits();
        return;
    }
    size_t remaining_bits = other.size() - rshift;
    bool bits_remain = true;
    
    for (auto block = other.blocks.begin() + 1; block != other.blocks.end(); block++)
    {
        Block new_block = zeros;
        new_block |= (*(block - 1) >> rshift);
        new_block |= (*block << lshift);
        blocks.push_back(new_block);
        if((remaining_bits <= bits_per_block) && bits_remain)
            bits_remain = false; 
        remaining_bits -= bits_per_block;        
    }
    if(bits_remain)
        blocks.push_back(*(other.blocks.end() - 1) >> rshift);
    zero_unused_bits();
    assert((size_t) compute_num_blocks(num_bits) == blocks.size());
}

template <typename Block>
inline bool DynamicBitset<Block>::operator<(const DynamicBitset<Block> &other) const
{
        
    size_t asize(size());
    size_t bsize(other.size());

    const DynamicBitset &a = *this;
    const DynamicBitset &b = other;

    if (!bsize)
    {
        return false;
    }
    else if (!asize)
    {
        return true;
    }
    else if (asize == bsize)
    {
        const vector<Block> &ablocks = blocks;
        const vector<Block> &bblocks = other.blocks;
        for (size_t i = 0; i < blocks.size(); i++)
        {
            if (ablocks[i] != bblocks[i])
            {
                size_t start_bit = i * bits_per_block;
                size_t end_bit = start_bit + bits_per_block;
                for (size_t j = start_bit; j < end_bit; j++)
                {
                    if (a[j] < b[j])
                        return true;
                    else if (a[j] > b[j])
                        return false;
                }
            }
        }
        return false;
    }
    else
    {
        size_t ia = 0;
        size_t ib = 0;
        if(asize > bsize){
            for (; ia < asize - bsize; ++ia)
            {
                if (a[ia])
                    return true;
            }  
        } else {
            for (; ib < bsize - asize; ++ib)
            {
                if (b[ib])
                    return true;
            }
        }
        for (; (ia < asize); ++ia, ++ib)
        {
            if (a[ia] < b[ib])
                return true;
            else if (a[ia] > b[ib])
                return false;
        }
        return false;
    }
}

template <typename Block>
inline size_t DynamicBitset<Block>::popcount(size_t to, size_t from) const
{
	assert(to <= num_bits);
	assert(to >= from);
    size_t from_bit = bit_index(from);
    size_t from_block = block_index(from);
	if ((bits_per_block - from_bit) > to - from) {
		Block block = blocks[from_block];
        Block mask = bit_mask(from_bit, bit_index(to));
		return __builtin_popcount(block & mask);
	}

    size_t count = 0;
    if(from_bit){
        Block block = blocks[from_block];
        Block mask = bit_mask(from_bit, bits_per_block);
        count = __builtin_popcount(block & mask);
        ++from_block;
    }

    size_t to_block = block_index(to);
    size_t to_bit = bit_index(to);

    count += popcnt(blocks.data() + from_block, to_block - from_block);
    
    if(to_bit){
        Block block = blocks[to_block];
        Block mask = bit_mask(0, to_bit);
        count += __builtin_popcount(block & mask);
    }
	
	return count;
}
}

/*
This source file was derived from the boost::dynamic_bitset library
version 1.54. Original copyright statement and license for this
original source follow.

Copyright (c) 2001-2002 Chuck Allison and Jeremy Siek
Copyright (c) 2003-2006, 2008 Gennaro Prota

Distributed under the Boost Software License, Version 1.0.

Boost Software License - Version 1.0 - August 17th, 2003

Permission is hereby granted, free of charge, to any person or organization
obtaining a copy of the software and accompanying documentation covered by
this license (the "Software") to use, reproduce, display, distribute,
execute, and transmit the Software, and to prepare derivative works of the
Software, and to permit third-parties to whom the Software is furnished to
do so, all subject to the following:

The copyright notices in the Software and this entire statement, including
the above license grant, this restriction and the following disclaimer,
must be included in all copies of the Software, in whole or in part, and
all derivative works of the Software, unless such copies or derivative
works are solely in the form of machine-executable object code generated by
a source language processor.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE, TITLE AND NON-INFRINGEMENT. IN NO EVENT
SHALL THE COPYRIGHT HOLDERS OR ANYONE DISTRIBUTING THE SOFTWARE BE LIABLE
FOR ANY DAMAGES OR OTHER LIABILITY, WHETHER IN CONTRACT, TORT OR OTHERWISE,
ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
DEALINGS IN THE SOFTWARE.
*/

#endif
