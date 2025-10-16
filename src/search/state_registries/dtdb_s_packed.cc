#include "dtdb_s_packed.h"

#include "../per_state_information.h"
#include "../task_proxy.h"

#include "../task_utils/task_properties.h"
#include "../utils/logging.h"

#include <iostream>

using namespace std;

DTDB_S_PackedStateRegistry::DTDB_S_PackedStateRegistry(const TaskProxy &task_proxy)
    : IStateRegistry(task_proxy), state_packer(task_properties::g_state_packers[task_proxy]),
      axiom_evaluator(g_axiom_evaluators[task_proxy]),
      num_variables(task_proxy.get_variables().size()) {


    State::get_variable_value = [this](const StateID& id) {
            static thread_local std::vector<PackedStateBin> s_buffer;
            s_buffer.clear();
            valla::read_sequence(valla::Slot<PackedStateBin>(root_backward[id.value], get_bins_per_state()), tree_table, std::back_inserter(s_buffer));

            std::vector<int> state_data(num_variables);
            for (int i = 0; i < num_variables; ++i) {
                state_data[i] = state_packer.get(s_buffer.data(), i);
            }

            return state_data;
    };

}

State DTDB_S_PackedStateRegistry::lookup_state(StateID id) const {
    return task_proxy.create_state(*this, id);
}

State DTDB_S_PackedStateRegistry::lookup_state(
    StateID id, vector<int> &&state_values) const {
    return task_proxy.create_state(*this, id, move(state_values));
}

const State &DTDB_S_PackedStateRegistry::get_initial_state() {
    if (!cached_initial_state) {
        State initial_state = task_proxy.get_initial_state();

        std::vector<PackedStateBin> buffer(get_bins_per_state());
        auto &tmp = initial_state.get_unpacked_values();
        for (auto i = 0; i < num_variables; ++i) {
            state_packer.set(buffer.data(), i, tmp[i]);
        }

        const auto root = valla::insert_sequence(buffer, tree_table);
        const auto [iter, success] = root_forward.emplace(root.i1, root_forward.size());
        const auto index = iter->second;
        if (success) {
            root_backward.push_back(root.i1);
            ++_registered_states;
        }

        // std::cout << "Insert: " << root.i1 << " " << index << " " << get_bins_per_state() << std::endl;

        StateID id = StateID(index);
        cached_initial_state = make_unique<State>(lookup_state(id));

        cached_initial_state->unpack();
    }
    return *cached_initial_state;
}

State DTDB_S_PackedStateRegistry::get_successor_state(const State &predecessor, const OperatorProxy &op) {
    assert(!op.is_axiom());
    /*
      TODO: ideally, we would not modify state_data_pool here and in
      insert_id_or_pop_state, but only at one place, to avoid errors like
      buffer becoming a dangling pointer. This used to be a bug before being
      fixed in https://issues.fast-downward.org/issue1115.
    */

    static thread_local std::vector<PackedStateBin> s_buffer;
    s_buffer.clear();
    valla::read_sequence(valla::Slot<PackedStateBin>(root_backward[predecessor.get_id().value], get_bins_per_state()), tree_table, std::back_inserter(s_buffer));

    /* Experiments for issue348 showed that for tasks with axioms it's faster
       to compute successor states using unpacked data. */
    if (task_properties::has_axioms(task_proxy)) {
        predecessor.unpack();
        vector<int> new_values = predecessor.get_unpacked_values();
        for (EffectProxy effect : op.get_effects()) {
            if (does_fire(effect, predecessor)) {
                FactPair effect_pair = effect.get_fact().get_pair();
                new_values[effect_pair.var] = effect_pair.value;
            }
        }
        axiom_evaluator.evaluate(new_values);
        for (size_t i = 0; i < new_values.size(); ++i) {
            state_packer.set(s_buffer.data(), i, new_values[i]);
        }

        const auto root = valla::insert_sequence(s_buffer, tree_table);
        const auto [iter, success] = root_forward.emplace(root.i1, root_forward.size());
        const auto index = iter->second;
        if (success) {
            root_backward.push_back(root.i1);
            ++_registered_states ;
        }

        return lookup_state(StateID(index), move(new_values));
    } else {
        for (EffectProxy effect : op.get_effects()) {
            if (does_fire(effect, predecessor)) {
                FactPair effect_pair = effect.get_fact().get_pair();
                state_packer.set(s_buffer.data(), effect_pair.var, effect_pair.value);
            }
        }

        const auto root = valla::insert_sequence(s_buffer, tree_table);
        const auto [iter, success] = root_forward.emplace(root.i1, root_forward.size());
        const auto index = iter->second;
        if (success) {
            root_backward.push_back(root.i1);
            ++_registered_states ;
        }

        return lookup_state(StateID(index));
    }
}


int DTDB_S_PackedStateRegistry::get_state_size_in_bytes() const {
    return get_bins_per_state() * sizeof(unsigned);
}

int DTDB_S_PackedStateRegistry::get_bins_per_state() const {
    return state_packer.get_num_bins();
}
void DTDB_S_PackedStateRegistry::print_statistics(utils::LogProxy &log) const {

    log << "Number of registered states: " << _registered_states << endl;
    log << "Closed list load factor: " << tree_table.size() << endl;
    log << "State size in bytes: " << get_state_size_in_bytes() << endl;
    log << "State set size: " << tree_table.mem_usage() / 1024 << " KB" << endl;

}