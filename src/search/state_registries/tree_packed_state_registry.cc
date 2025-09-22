#include "tree_packed_state_registry.h"

#include "../per_state_information.h"
#include "../task_proxy.h"

#include "../task_utils/task_properties.h"
#include "../utils/logging.h"

#include <iostream>

using namespace std;

TreePackedStateRegistry::TreePackedStateRegistry(const TaskProxy &task_proxy)
    : IStateRegistry(task_proxy), state_packer(task_properties::g_state_packers[task_proxy]),
      axiom_evaluator(g_axiom_evaluators[task_proxy]),
      num_variables(task_proxy.get_variables().size()) {

    State::get_variable_value = [this](const StateID& id) {
            std::vector<uint32_t> buffer = tree_table.lookup(id.value);

            std::vector<int> state_data(num_variables);
            for (int i = 0; i < num_variables; ++i) {
                state_data[i] = state_packer.get(buffer.data(), i);
            }

            return std::vector<int>{state_data.begin(), state_data.end()};
    };
}

StateID TreePackedStateRegistry::insert_id_or_pop_state() {
    return StateID(0);
}

State TreePackedStateRegistry::lookup_state(StateID id) const {
    std::vector<uint32_t> buffer = tree_table.lookup(id.value);

    std::vector<int> state_values(num_variables);
    for (int i = 0; i < num_variables; ++i) {
        state_values[i] = state_packer.get(buffer.data(), i);
    }

    return task_proxy.create_state(*this, id, move(state_values));
}

State TreePackedStateRegistry::lookup_state(
    StateID id, vector<int> &&state_values) const {
    return task_proxy.create_state(*this, id, move(state_values));
}

const State &TreePackedStateRegistry::get_initial_state() {
    if (!cached_initial_state) {
        State initial_state = task_proxy.get_initial_state();

        std::vector<uint32_t> buffer(get_bins_per_state());
        auto &tmp = initial_state.get_unpacked_values();
        for (auto i = 0; i < num_variables; ++i) {
            state_packer.set(buffer.data(), i, tmp[i]);
        }
        uint32_t index = tree_table.insert(buffer);
        ++_registered_states;
        StateID id = StateID(index);
        cached_initial_state = make_unique<State>(lookup_state(id));

        cached_initial_state->unpack();
    }
    return *cached_initial_state;
}

State TreePackedStateRegistry::get_successor_state(const State &predecessor, const OperatorProxy &op) {
    assert(!op.is_axiom());

    predecessor.unpack();
    auto& tmp = predecessor.get_unpacked_values();
    std::vector<uint32_t> new_state_values(tmp.begin(), tmp.end());

    for (EffectProxy effect : op.get_effects()) {
        if (does_fire(effect, predecessor)) {
            FactPair effect_pair = effect.get_fact().get_pair();
            new_state_values[effect_pair.var] = effect_pair.value;
        }
    }

    if (task_properties::has_axioms(task_proxy))
        axiom_evaluator.evaluate(reinterpret_cast<std::vector<int> &>(new_state_values));

    std::vector<uint32_t> buffer(get_bins_per_state());
    for (auto i = 0; i < num_variables; ++i) {
        state_packer.set(buffer.data(), i, new_state_values[i]);
    }

    uint32_t index = tree_table.insert(buffer);
    // For now, assume all inserts create new states
    // This is a simplification - in production, you'd need proper duplicate detection
    ++_registered_states;

    return lookup_state(StateID(index), {new_state_values.begin(), new_state_values.end()});
}

int TreePackedStateRegistry::get_state_size_in_bytes() const {
    return get_bins_per_state() * sizeof(uint32_t);
}

int TreePackedStateRegistry::get_bins_per_state() const {
    return state_packer.get_num_bins();
}

void TreePackedStateRegistry::print_statistics(utils::LogProxy &log) const {
    log << "Number of registered states: " << _registered_states << endl;
    log << "Closed list load factor: " << tree_table.size() << endl;
    log << "State size in bytes: " << get_state_size_in_bytes() << endl;
    log << "State set size: " << tree_table.mem_usage() / 1024 << " KB" << endl;
}