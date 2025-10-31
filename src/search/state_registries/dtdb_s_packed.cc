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
    size_t usage = 0;
    usage += tree_table.mem_usage();
    usage += root_forward.capacity() * sizeof(PackedStateBin);
    usage += root_backward.mem_usage();
    return usage;
}

int DTDB_S_PackedStateRegistry::get_bins_per_state() const {
    return state_packer.get_num_bins();
}
void DTDB_S_PackedStateRegistry::print_statistics(utils::LogProxy &log) const {
    // Avg bins per state
    log << "Number of registered states: " << _registered_states << endl;
    log << "Entries in state set: " << tree_table.size() << endl;
    const int bins_per_entry = 2;
    log << "Bins per entry: " << bins_per_entry << endl;
    log << "Average bins per state: " << (static_cast<double>(tree_table.size()) * bins_per_entry / _registered_states) << endl;

    // State set size
    log << "State set size: " << tree_table.mem_usage() << " B" << endl;
    log << "Lookup structure size: " << root_backward.mem_usage() +
        (root_forward.capacity() * (sizeof(PackedStateBin))) << " B" << endl;
    log << "State registry size: " << get_state_size_in_bytes() << " B" << endl;

    // State size in bins
    log << "Number of bins in state: " << get_bins_per_state() << endl;

    // Number of bins the operators touch
    log << "Number of operators: " << task_proxy.get_operators().size() << endl;

    int num_fluents = 0;
    int num_derived = 0;
    for (const auto& var : task_proxy.get_variables()) {
        num_fluents += !var.is_derived();
        num_derived += var.is_derived();
    }
    log << "Number of fluents: " << num_fluents << endl;
    log << "Number of derived: " << num_derived << endl;

    log << "Number of operator touches nodes: [";
    uint32_t touches = 0;
    for (const auto& op : task_proxy.get_operators()) {
        unordered_set<int> touched_bins;
        for (const auto& eff : op.get_effects()) {
            const int var_id = eff.get_fact().get_variable().get_id();
            const int bin_id = state_packer.get_bin_index(var_id) / 2;
            touched_bins.insert(bin_id);
        }
        touches += touched_bins.size();
        log << touched_bins.size() << ", ";
    }
    log << "]" << endl;
    log << "Total number of operator touches: " << touches << endl;
    log << "Average number of operator touches: " << static_cast<double>(touches) / task_proxy.get_operators().size() << endl;


}