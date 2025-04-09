#include "bdd_closed_list.h"

#include "../utils/logging.h"
#include "../plugins/plugin.h"


using namespace std;

namespace bdd_closed_list {
BddClosedList::BddClosedList(
    const int cache_size, 
    const bool full_print,
    const bool gamer_ordering,
    const bool dynamic_ordering) : sym_vars(gamer_ordering, dynamic_ordering, tasks::g_root_task, cache_size), full_print(full_print) {
    sym_vars.init();
    closed_list = sym_vars.zeroBDD();
    // Fast evaluation
    bin_state.resize(sym_vars.getNumBDDVars(), 0);
    var_order = sym_vars.getVarOrder();
}

int *BddClosedList::get_binary_description(const State &state) const {
    int pos = 0;
    const vector<int> &state_values = state.get_unpacked_values();
    for (int v : var_order) {
        for (size_t j = 0; j < sym_vars.vars_index_pre(v).size(); j++) {
            bin_state[pos++] = ((state_values[v] >> j) % 2);
            bin_state[pos++] = 0;         //Skip interleaving variable
        }
    }
    return &(bin_state[0]);
}


void BddClosedList::add_state(const State &state) {
    closed_list += sym_vars.getStateBDD(state);
    ++size;
}

// TODO: We can probably do this much more efficient by travesing the BDD according to the values
bool BddClosedList::contains_state(const State &state) {
    BDD intersection = closed_list * sym_vars.getStateBDD(state);
    // int *inputs = get_binary_description(state);
    // BDD intersection2 = closed_list.Eval(inputs);
    // assert(intersection.IsZero() == intersection2.IsZero());
    return !intersection.IsZero();
}

void BddClosedList::print() const {
    cout << "Closed list contains " << size << " states." << endl;
    if (full_print)
        sym_vars.to_dot(closed_list, "closed_list.dot");
}


void BddClosedList::add_options_to_parser(
    plugins::Feature &feature) {
    feature.add_option<int>(
        "cache_size",
        "the cache size used for the BDD",
        "16000000");
    feature.add_option<bool>(
        "full_print",
        "print",
        "false");
}

class BddClosedListFeature
    : public plugins::TypedFeature<ClosedList, BddClosedList> {
public:
    BddClosedListFeature() : TypedFeature("bdd") {
        document_title("BDDBased Closed List");

        BddClosedList::add_options_to_parser(*this);
        symbolic::SymVariables::add_options_to_parser(*this);
    }

    virtual shared_ptr<BddClosedList>
    create_component(const plugins::Options &opts) const override {
        
        return plugins::make_shared_from_arg_tuples<BddClosedList>(
            opts.get<int>("cache_size", 16000000L),
            opts.get<bool>("full_print"),
            opts.get<bool>("gamer_ordering"),
            opts.get<bool>("dynamic_reordering")
        );
    }
};


static plugins::FeaturePlugin<BddClosedListFeature> _plugin;
}
