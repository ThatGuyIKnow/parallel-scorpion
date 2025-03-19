#ifndef LOES_LOES_H
#define LOES_LOES_H
#include <list>
#include <vector>
#include "../algorithms/dynamic_bitset.h"
#include "bitstate.h"

using namespace std;
namespace loes{
class BitSeq {
unique_ptr<dynamic_bitset::DynamicBitset<unsigned char>> data;
size_t m_size = 0;
public:
    BitSeq() {
        data = make_unique<dynamic_bitset::DynamicBitset<unsigned char>>();
    }
    void reset() {
        data.reset();
        m_size = 0;
    }
    void resize(size_t new_size) {
        assert(new_size <= m_size);
        m_size = new_size;
    }
    bool operator[](size_t idx) const {
        return data->operator[](idx);
    }
    void set(size_t pos, bool val = true) {
        data->set(pos, val);
    }
    void set_back(bool val) {
        data->set(m_size - 1, val);
    }
    size_t popcount(size_t to, size_t from = 0) const {
        return data->popcount(to, from);
    }
    bool pop_back() {
        --m_size;
        return data->operator[](m_size);
    }
    void push_back(bool val) {
        if(m_size == data->size())
            data->push_back(val);
        else
            data->set(m_size, val);
        ++m_size;
    }
    void push_back(const BitSeq &other) {
        data->resize(m_size);
        m_size += other.size();
        other.data->resize(other.m_size);
        data->push_back(*other.data);
    }
    size_t size() const {
        return m_size;
    }
};
//using BitSeq = dynamic_bitset::DynamicBitset<unsigned char>;

class GenericLoes
{
protected:
    struct Data {
        shared_ptr<vector<BitSeq>> tree_levels;
        BitSeq combined_tree_levels;
        vector<size_t> level_offsets = { 0 };
        size_t state_bit_length = 0;
        static const short LOG2_BLOCK_SIZE = 16; //Block size = 2^16
        static const short LOG2_SUBBLOCK_SIZE = 9; //Subblock size = 2^9
        vector<unsigned long long> block_popcount = { 0 }; 
        vector<unsigned short> subblock_popcount = { 0 };
        bool is_empty = true;
        bool is_compiled = false;
        bool has_combined_tree_levels = false;
        bool level_offsets_computed = false;
        bool reuse_tree_levels = false;
        
        void subblock_popcount_push_back(size_t popcount);
        bool operator[](size_t idx) const;
        size_t rank(size_t offset) const;
        size_t size() const;
        size_t getLevelWithOffset(size_t offset) const;
        size_t level_size(size_t level) const;
        size_t get(size_t level, size_t idx) const;
        void compile_combined_treelevels();
        void compute_level_offsets();
        Data(bool reuse_tree_levels = false) : reuse_tree_levels(reuse_tree_levels) {}
    };

    mutable Data data = {};
    mutable unique_ptr<BitState> prev_state;
    template<class BitStateT>
    void add_state_with_type(const BitStateT &state);
    template<class LoesT>
    void merge(const LoesT &loes1, const LoesT &loes2, LoesT &new_loes);
    void update_prev_state(const SequentialBitState &state);
    void update_prev_state(const BitState &state);
    virtual size_t path_offset(const BitState &state) const;
    virtual void add_state(const SequentialBitState &state);
    virtual void add_state(const BitState &state);
    bool can_reuse_tree_levels() const { return data.reuse_tree_levels && data.is_compiled; } 
public:
    GenericLoes(bool reuse_tree_levels = false) : data(reuse_tree_levels) {}
    virtual ~GenericLoes() = default;

    size_t member_index(const BitState &state) const;
    bool contains(const BitState &state) const;
    virtual size_t size() const;
    size_t bit_size() const;
    virtual void print(bool full = false) const; //TODO: remove later, real LOES to big to print
    void compile() const;
    struct iterator
    {
        friend class Loes;
        friend class GenericLoes;
        SequentialBitState operator*() const {
            SequentialBitState s(loes->data.state_bit_length);
            deref_into(s);
            return s;
        }
        virtual iterator& operator++();
        virtual iterator operator++(int);
        friend bool operator==(const iterator& a, const iterator& b);
        friend bool operator!=(const iterator& a, const iterator& b);
        virtual ~iterator() = default;

    protected:
        const GenericLoes* loes;
        bool is_end;
        bool combined_tree_levels;
        vector<size_t> offset_state;
        size_t min_updated_level = 0;
        virtual void deref_into(SequentialBitState &state) const;
        virtual size_t incr_seperate_levels();
        virtual size_t incr_combined_levels();
        iterator(const GenericLoes* loes, vector<size_t> offset_state, bool combined_tree_levels) : loes(loes), is_end(false), combined_tree_levels(combined_tree_levels),
                offset_state(offset_state) {}
        iterator(const GenericLoes* loes) : loes(loes), is_end(true) {}
    };
};
class Loes : public GenericLoes
{    
public:
    Loes() {}
    Loes(const BitState &state, bool will_reuse_tree_levels = false);
    Loes(const BitState &state1, const BitState &state2, bool will_reuse_tree_levels = false);
    Loes(const Loes &loes1, const Loes &loes2, bool reuse_tree_levels = false) : GenericLoes(reuse_tree_levels) { merge(loes1, loes2, *this); }
    
    iterator begin() const;
    iterator end() const;
};
}

#endif