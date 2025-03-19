#include "vector_state_registry.h"

#include "../per_state_information.h"
#include "../task_proxy.h"

#include "../task_utils/task_properties.h"
#include "../utils/logging.h"

using namespace std;

VectorStateRegistry::VectorStateRegistry(const TaskProxy &task_proxy)
    : StateRegistry(task_proxy),
      state_data_pool(get_bins_per_state()),
      registered_states(
          StateIDSemanticHash(state_data_pool, get_bins_per_state()),
          StateIDSemanticEqual(state_data_pool, get_bins_per_state())) {
}

StateID VectorStateRegistry::insert_id_or_pop_state() {
    StateID id(state_data_pool.size() - 1);
    auto result = registered_states.insert(id.value);
    bool is_new_entry = result.second;
    if (!is_new_entry) {
        state_data_pool.pop_back();
    }
    assert(registered_states.size() == state_data_pool.size());
    return StateID(result.first);
}

StateID VectorStateRegistry::insert_buffered_state(const PackedStateBin *buffer) {
    state_data_pool.push_back(buffer);
    return insert_id_or_pop_state();
}

State VectorStateRegistry::lookup_state(StateID id) const {
    const PackedStateBin *buffer = state_data_pool[id.value];
    return task_proxy.create_state(&*this, id, buffer);
}

State VectorStateRegistry::lookup_state(StateID id, vector<int> &&state_values) const {
    const PackedStateBin *buffer = state_data_pool[id.value];
    return task_proxy.create_state(&*this, id, buffer, std::move(state_values));
}

State VectorStateRegistry::get_successor_state(const State &predecessor, const OperatorProxy &op) {
    assert(!op.is_axiom());
    state_data_pool.push_back(predecessor.get_buffer());
    PackedStateBin *buffer = state_data_pool[state_data_pool.size() - 1];
    if (task_properties::has_axioms(task_proxy)) {
        predecessor.unpack();
        std::vector<int> new_values = predecessor.get_unpacked_values();
        for (EffectProxy effect : op.get_effects()) {
            if (does_fire(effect, predecessor)) {
                FactPair effect_pair = effect.get_fact().get_pair();
                new_values[effect_pair.var] = effect_pair.value;
            }
        }
        axiom_evaluator.evaluate(new_values);
        for (size_t i = 0; i < new_values.size(); ++i) {
            state_packer.set(buffer, i, new_values[i]);
        }
        StateID id = insert_id_or_pop_state();
        return lookup_state(id, std::move(new_values));
    } else {
        for (EffectProxy effect : op.get_effects()) {
            if (does_fire(effect, predecessor)) {
                FactPair effect_pair = effect.get_fact().get_pair();
                state_packer.set(buffer, effect_pair.var, effect_pair.value);
            }
        }
        StateID id = insert_id_or_pop_state();
        return lookup_state(id);
    }
}

size_t VectorStateRegistry::size() const {
    return registered_states.size();
}

const State &VectorStateRegistry::get_initial_state() {
    if (!cached_initial_state) {
        int num_bins = get_bins_per_state(state_packer);
        std::unique_ptr<PackedStateBin[]> buffer(new PackedStateBin[num_bins]);
        std::fill_n(buffer.get(), num_bins, 0);
        State initial_state = task_proxy.get_initial_state();
        for (size_t i = 0; i < initial_state.size(); ++i) {
            state_packer.set(buffer.get(), i, initial_state[i].get_value());
        }
        state_data_pool.push_back(buffer.get());
        StateID id = insert_id_or_pop_state();
        cached_initial_state = std::make_unique<State>(lookup_state(id));
    }
    return *cached_initial_state;
}

void VectorStateRegistry::print_statistics(utils::LogProxy &log) const {
    log << "Number of registered states: " << size() << std::endl;
    log << "Closed list load factor: " << registered_states.size();
}