#ifndef CLOSED_LISTS_BDD_CLOSED_LIST_H
#define CLOSED_LISTS_BDD_CLOSED_LIST_H

#include "../closed_list.h"

#include "../symbolic/sym_variables.h"


namespace bdd_closed_list {
class BddClosedList : public ClosedList  {
    symbolic::SymVariables sym_vars;
    BDD closed_list;
    size_t size = 0;
    bool full_print;
    mutable std::vector<int> bin_state;
    std::vector<int> var_order;

    int *get_binary_description(const State &state) const;
public:
    BddClosedList(const options::Options &opts);
    virtual void add_state(const State &state) override;
    virtual bool contains_state(const State &state) const override;
    virtual void print() const override;
};
}

#endif
