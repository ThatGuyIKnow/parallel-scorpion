#include "tree_unpacked_state_registry.h"

#include "../per_state_information.h"
#include "../task_proxy.h"
#include "../task_utils/task_properties.h"
#include "../utils/logging.h"

#include <iostream>

using namespace std;

TreeUnpackedStateRegistry::TreeUnpackedStateRegistry(const TaskProxy &task_proxy)
    : IStateRegistry(task_proxy), state_packer(task_properties::g_state_packers[task_proxy]),
      axiom_evaluator(g_axiom_evaluators[task_proxy]),
      num_variables(task_proxy.get_variables().size()) {

    State::get_variable_value = [this](const StateID& id) {
        std::vector<int> state_data = tree_table.lookup(id.value);
        return state_data;
    };
}

StateID TreeUnpackedStateRegistry::insert_id_or_pop_state() {
    return StateID(0);
}

State TreeUnpackedStateRegistry::lookup_state(StateID id) const {
    std::vector<int> tmp = tree_table.lookup(id.value);
    return task_proxy.create_state(*this, id, move(tmp));
}

State TreeUnpackedStateRegistry::lookup_state(
    StateID id, vector<int> &&state_values) const {
    return task_proxy.create_state(*this, id, move(state_values));
}

const State &TreeUnpackedStateRegistry::get_initial_state() {
    if (!cached_initial_state) {
        State initial_state = task_proxy.get_initial_state();

        std::vector<int> tmp = initial_state.get_unpacked_values();
        uint32_t index = tree_table.insert(tmp);
        ++_registered_states;
        StateID id = StateID(index);
        cached_initial_state = make_unique<State>(lookup_state(id));

        cached_initial_state->unpack();
    }
    return *cached_initial_state;
}

State TreeUnpackedStateRegistry::get_successor_state(const State &predecessor, const OperatorProxy &op) {
    assert(!op.is_axiom());

    std::vector<int> successor_values{predecessor.get_unpacked_values()};

    for (EffectProxy effect : op.get_effects()) {
        if (does_fire(effect, predecessor)) {
            FactPair effect_pair = effect.get_fact().get_pair();
            successor_values[effect_pair.var] = effect_pair.value;
        }
    }

    if (task_properties::has_axioms(task_proxy))
        axiom_evaluator.evaluate(successor_values);

    uint32_t index = tree_table.insert(successor_values);
    // For now, assume all inserts create new states
    // This is a simplification - in production, you'd need proper duplicate detection
    ++_registered_states;

    return lookup_state(StateID(index), {successor_values.begin(), successor_values.end()});
}

int TreeUnpackedStateRegistry::get_state_size_in_bytes() const {
    return num_variables * sizeof(int);
}

int TreeUnpackedStateRegistry::get_bins_per_state() const {
    return state_packer.get_num_bins();
}

void TreeUnpackedStateRegistry::print_statistics(utils::LogProxy &log) const {
    log << "Number of registered states: " << _registered_states << endl;
    log << "Closed list load factor: " << tree_table.size() << endl;
    log << "State size in bytes: " << get_state_size_in_bytes() << endl;
    log << "State set size: " << tree_table.mem_usage() / 1024 << " KB" << endl;
}