#include "bdd_closed_list.h"

#include "../utils/logging.h"
#include "../option_parser.h"
#include "../plugin.h"


using namespace std;

namespace bdd_closed_list {
BddClosedList::BddClosedList(const options::Options &opts) : sym_vars(opts, tasks::g_root_task, opts.get<int>("cache_size", 16000000L)), full_print(opts.get<bool>("full_print")) {
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
bool BddClosedList::contains_state(const State &state) const {
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

static shared_ptr<ClosedList> _parse(OptionParser &parser) {
    parser.document_synopsis(
        "Closed list using BDD data structure",
        "");
    parser.add_option<int>(
        "cache_size",
        "the cache size used for the BDD",
        OptionParser::NONE);
    parser.add_option<bool>(
        "full_print",
        "will print() create a .dot file",
        "false");
    symbolic::SymVariables::add_options_to_parser(parser);

    ClosedList::add_options_to_parser(parser);
    Options opts = parser.parse();

    if (parser.dry_run()) {
        return nullptr;
    }

    return std::make_shared<BddClosedList>(opts);
}


static Plugin<ClosedList> _plugin("bdd", _parse);
}
