#include "unpacked.h"

#include "../per_state_information.h"
#include "../task_proxy.h"

#include "../task_utils/task_properties.h"
#include "../utils/logging.h"

using namespace std;

UnpackedStateRegistry::UnpackedStateRegistry(const TaskProxy &task_proxy)
    : IStateRegistry(task_proxy), state_packer(task_properties::g_state_packers[task_proxy]),
      axiom_evaluator(g_axiom_evaluators[task_proxy]),
      num_variables(task_proxy.get_variables().size()),
      state_data_pool(num_variables),
      registered_states(
          0,
          StateIDSemanticHash(state_data_pool, num_variables),
          StateIDSemanticEqual(state_data_pool, num_variables)) {

    State::get_variable_value =
        [this](const StateID& id) {
            std::vector<int> state_data(num_variables);
            const int *buffer = state_data_pool[id.value];
            for (int i = 0; i < num_variables; ++i) {
                state_data[i] = buffer[i];
            }
            return state_data;
        };
}

StateID UnpackedStateRegistry::insert_id_or_pop_state() {
    /*
      Attempt to insert a StateID for the last state of state_data_pool
      if none is present yet. If this fails (another entry for this state
      is present), we have to remove the duplicate entry from the
      state data pool.
    */
    StateID id(state_data_pool.size() - 1);
    auto result = registered_states.insert(id.value);
    bool is_new_entry = result.second;
    if (!is_new_entry) {
        state_data_pool.pop_back();
    }
    assert(registered_states.size() == state_data_pool.size());
    return StateID(*result.first);
}

State UnpackedStateRegistry::lookup_state(StateID id) const {
    return task_proxy.create_state(*this, id);
}

State UnpackedStateRegistry::lookup_state(
    StateID id, vector<int> &&state_values) const {
    return task_proxy.create_state(*this, id, std::move(state_values));
}

const State &UnpackedStateRegistry::get_initial_state() {
    if (!cached_initial_state) {

        State initial_state = task_proxy.get_initial_state();
        // unique_ptr<int[]> buffer(new int[num_variables]);
        // // Initialize all values to zero.
        // fill_n(buffer.get(), num_variables, 0);
        // for (size_t i = 0; i < initial_state.size(); ++i) {
        //     buffer[i] = initial_state[i].get_value();
        // }
        initial_state.unpack();
        state_data_pool.push_back(initial_state.get_unpacked_values().data());
        StateID id = insert_id_or_pop_state();
        cached_initial_state = make_unique<State>(lookup_state(id));
    }
    return *cached_initial_state;
}

//TODO it would be nice to move the actual state creation (and operator application)
//     out of the UnpackedStateRegistry. This could for example be done by global functions
//     operating on state buffers (int *).
State UnpackedStateRegistry::get_successor_state(const State &predecessor, const OperatorProxy &op) {
    assert(!op.is_axiom());
    /*
      TODO: ideally, we would not modify state_data_pool here and in
      insert_id_or_pop_state, but only at one place, to avoid errors like
      buffer becoming a dangling pointer. This used to be a bug before being
      fixed in https://issues.fast-downward.org/issue1115.
    */
    auto unpacked_pred = state_data_pool[predecessor.get_id().value];
    state_data_pool.push_back(unpacked_pred);
    int *buffer = state_data_pool[state_data_pool.size() - 1];

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
            buffer[i] = new_values[i];
        }
        /*
          NOTE: insert_id_or_pop_state possibly invalidates buffer, hence
          we use lookup_state to retrieve the state using the correct buffer.
        */
        StateID id = insert_id_or_pop_state();
        return lookup_state(id, move(new_values));
    } else {
        for (EffectProxy effect : op.get_effects()) {
            if (does_fire(effect, predecessor)) {
                FactPair effect_pair = effect.get_fact().get_pair();
                buffer[effect_pair.var] = effect_pair.value;
            }
        }
        StateID id = insert_id_or_pop_state();
        return lookup_state(id);
    }
}

int UnpackedStateRegistry::get_state_size_in_bytes() const {
    return num_variables * sizeof(int);
}

size_t UnpackedStateRegistry::get_memory_usage() const
{
    size_t usage = 0;

    usage += state_data_pool.capacity() * get_state_size_in_bytes();
    usage += registered_states.capacity() * (sizeof(int) + 1);

    return usage;
}

size_t UnpackedStateRegistry::get_occupied_memory_usage() const {
    size_t usage = 0;

    usage += state_data_pool.size() * get_state_size_in_bytes();
    usage += registered_states.size() * (sizeof(int) + 1);

    return usage;
}

void UnpackedStateRegistry::print_statistics(utils::LogProxy &log) const {
    // Avg bins per state
    log << "Number of registered states: " << registered_states.size() << endl;
    log << "Entries in state set: " << registered_states.size() << endl;
    const int bins_per_entry = num_variables;
    log << "Bins per entry: " << bins_per_entry << endl;
    log << "Average bins per state: " << bins_per_entry << endl;

    // State set size
    log << "State set size: " << get_memory_usage() << " B" << endl;
    log << "Lookup structure size: " << (registered_states.capacity() * (sizeof(int) + 1)) << " B" << endl;
    log << "State set size: " << get_state_size_in_bytes() << " B" << endl;

    // State size in bins
    log << "Number of bins in state: " << num_variables << endl;

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
            const int bin_id = var_id / 2;
            touched_bins.insert(bin_id);
        }
        touches += touched_bins.size();
        log << touched_bins.size() << ", ";
    }
    log << "]" << endl;
    log << "Total number of operator touches: " << touches << endl;
    log << "Average number of operator touches: " << static_cast<double>(touches) / task_proxy.get_operators().size() << endl;
}

