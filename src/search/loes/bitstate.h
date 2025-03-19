#ifndef LOES_BITSTATE_H
#define LOES_BITSTATE_H
#include <vector>
#include <iostream>
#include "../algorithms/dynamic_bitset.h"

using namespace std;

namespace loes{
class SequentialBitState;


class BitState {
friend class SequentialBitState;
virtual bool equals(const BitState& other) const {
    if(size() != other.size())
        return false;
    for (size_t i = 0; i < size(); i++)
    {
        if(operator[](i) != other[i])
            return false;
    }
    return true;
}
virtual bool less_than(const BitState& other) const {
    if(size() != other.size())
        return false;
    
    for (size_t i = 0; i < size(); i++)
    {
        if(operator[](i) != other[i])
            return operator[](i) < other[i];
    }
    return false;
}

public:
virtual ~BitState() {}
virtual size_t size() const = 0;
virtual void set(size_t pos, bool val = true) = 0;
virtual unique_ptr<BitState> clone() const = 0;

virtual size_t first_differing_bit(const BitState &other) const {
    if(size() != other.size())
        return size();
    
    for (size_t i = 0; i < size(); i++)
    {
        if(operator[](i) != other[i])
            return i;
    }
    return size();
}


virtual bool operator[](size_t pos) const = 0;

bool operator==(const BitState &other) const { return equals(other); }
bool operator!=(const BitState &other) const { return !(*this == other); }
bool operator<(const BitState &other) const { return less_than(other); }
bool operator>(const BitState& other) const { return other < *this; }
bool operator>=(const BitState& other) const { return !(*this < other); }
bool operator<=(const BitState& other) const { return !(*this > other); }
};





class SequentialBitState : public BitState {
using Data = dynamic_bitset::DynamicBitset<unsigned char>;
Data data;

virtual bool equals(const BitState& other) const override {
    if(SequentialBitState const* p = dynamic_cast<SequentialBitState const*>(&other)){
        return data == p->data;
    }
    return BitState::equals(other);
}

virtual bool less_than(const BitState& other) const {
    if(SequentialBitState const* p = dynamic_cast<SequentialBitState const*>(&other)){
        return data < p->data;
    }
    return BitState::less_than(other);
}


public:
SequentialBitState(size_t num_bits) : data(num_bits) {}
SequentialBitState() : SequentialBitState(0) {}
virtual size_t size() const override { return data.size(); }
virtual void set(size_t pos, bool val = true) override { data.set(pos, val); }
virtual unique_ptr<BitState> clone() const override {
    return make_unique<SequentialBitState>(*this);
}

virtual size_t first_differing_bit(const BitState &other) const override {
    if(SequentialBitState const* p = dynamic_cast<SequentialBitState const*>(&other)){
        return data.first_differing_bit(p->data);
    }
    return BitState::first_differing_bit(other);
}

size_t first_differing_bit(const SequentialBitState &other) const {
    return data.first_differing_bit(other.data);
}

virtual bool operator[](size_t pos) const override { return data[pos]; }

bool operator==(const SequentialBitState &other) const { return data == other.data; }
bool operator<(const SequentialBitState &other) const { return data < other.data; }
bool operator>(const SequentialBitState& other) const { return other < *this; }
bool operator>=(const SequentialBitState& other) const { return !(*this < other); }
bool operator<=(const SequentialBitState& other) const { return !(*this > other); }

SequentialBitState& operator++() {
    data++;
    return *this;
}
SequentialBitState operator++(int){
    SequentialBitState tmp = *this;
    ++(*this);
    return tmp;
}

};


class BitMap {
    vector<size_t> block_indices;
    vector<unsigned int> bit_indices;
    size_t num_bits;
    public:
    BitMap(const vector<unsigned int> &var_bit_lengths){
        num_bits = 0;
        for (size_t block_i = 0; block_i < var_bit_lengths.size(); block_i++)
        {
            for (unsigned int bit_i = 0; bit_i < var_bit_lengths[block_i]; bit_i++)
            {
                bit_indices.push_back(bit_i);
                block_indices.push_back(block_i);
                ++num_bits;
            }
        }
    }
    BitMap(const vector<unsigned int> &var_bit_lengths, vector<size_t> bit_order){
        num_bits = bit_order.size();
        block_indices.resize(num_bits);
        bit_indices.resize(num_bits);
        size_t i = 0;
        for (size_t block_i = 0; block_i < var_bit_lengths.size(); block_i++)
        {
            for (unsigned int bit_i = 0; bit_i < var_bit_lengths[block_i]; bit_i++)
            {
                bit_indices[bit_order[i]] = bit_i;
                block_indices[bit_order[i]] = block_i;
                ++i;
            }
        }
    }
    size_t block_index(size_t idx) const {
        return block_indices[idx];
    }
    unsigned int bit_index(size_t idx) const {
        return bit_indices[idx];
    }
    size_t size() const {
        return num_bits;
    }
};

class MappedBitState : public BitState {
using Data = vector<int>;
Data data;
const shared_ptr<BitMap> bitmap;
size_t num_bits;

int bit_mask(size_t pos) const {
    return 1 << bitmap->bit_index(pos);
}

public:
MappedBitState(vector<int> data, const shared_ptr<BitMap> bitmap) : data(data), bitmap(bitmap), num_bits(bitmap->size()) {}
MappedBitState() {}

virtual size_t size() const override { return num_bits; }
virtual void set(size_t pos, bool val = true) override { 
    assert(pos < num_bits);
    if(val)
        data[bitmap->block_index(pos)] |= bit_mask(pos);
    else
        data[bitmap->block_index(pos)] &= ~bit_mask(pos);
}
virtual unique_ptr<BitState> clone() const override {
    return make_unique<MappedBitState>(*this);
}

virtual bool operator[](size_t pos) const override {
    assert(pos < num_bits);
    return (data[bitmap->block_index(pos)] & bit_mask(pos)) != 0;
}

};
}
#endif