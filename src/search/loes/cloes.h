#ifndef LOES_CLOES_H
#define LOES_CLOES_H
#include "loes.h"
#include <vector>

using namespace std;

namespace loes{
class Cloes : public GenericLoes{
    friend class GenericLoes;
    size_t m_size = 0;

    virtual size_t path_offset(const BitState &state) const override;
    template<class BitStateT>
    void add_state_with_type(const BitStateT &state);
    virtual void add_state(const SequentialBitState &state) override;
    virtual void add_state(const BitState &state) override;
public:
    Cloes() {}
    Cloes(const BitState &state, bool reuse_tree_levels = false);
    Cloes(const BitState &state1, const BitState &state2, bool will_reuse_tree_levels = false);
    Cloes(const Cloes &loes1, const Cloes &loes2, bool reuse_tree_levels = false) : GenericLoes(reuse_tree_levels) { merge(loes1, loes2, *this); }

    virtual void print(bool full = false) const override;
    virtual size_t size() const override;

    struct iterator : public Loes::iterator
    {
        friend class Cloes;
        friend class GenericLoes;
        virtual iterator& operator++() override;
        virtual Loes::iterator operator++(int) override;
    private:
        const Cloes* cloes;
        bool found_comp = false;
        size_t comp_level = cloes->data.state_bit_length;
        size_t comp_count = 0;
        size_t end_count = 0;
        SequentialBitState comp_count_state;
        
        iterator(const Cloes* cloes, vector<size_t> offset_state, bool combined_tree_levels) : Loes::iterator(cloes, offset_state, combined_tree_levels) , cloes(cloes),
        comp_count_state(cloes->data.state_bit_length){}
        iterator(const Cloes* cloes) : Loes::iterator(cloes), cloes(cloes) {}
        iterator(const Cloes* cloes, vector<size_t> offset_state, size_t comp_level, bool combined_tree_levels) : Cloes::iterator(cloes, offset_state, combined_tree_levels){ 
            report_compression(comp_level);
        }
        
        void report_compression(size_t level);
        virtual void deref_into(SequentialBitState &state) const override;
        virtual size_t incr_seperate_levels() override;
        virtual size_t incr_combined_levels() override;
    };
    iterator begin() const;
    iterator end() const;
};
}

#endif