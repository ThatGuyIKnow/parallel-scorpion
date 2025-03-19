#include "state_registry.h"

#include "../per_state_information.h"
#include "../task_proxy.h"

#include "../task_utils/task_properties.h"
#include "../utils/logging.h"
#include <memory>

using namespace std;

StateRegistry::StateRegistry(const TaskProxy &task_proxy)
    : task_proxy(task_proxy),
      state_packer(task_properties::g_state_packers[task_proxy]),
      axiom_evaluator(g_axiom_evaluators[task_proxy]),
      num_variables(task_proxy.get_variables().size()){
}

const State &StateRegistry::get_initial_state() {
    if (!cached_initial_state) {
        int num_bins = get_bins_per_state();
        unique_ptr<PackedStateBin[]> buffer(new PackedStateBin[num_bins]);
        // Avoid garbage values in half-full bins.
        fill_n(buffer.get(), num_bins, 0);

        State initial_state = task_proxy.get_initial_state();
        for (size_t i = 0; i < initial_state.size(); ++i) {
            state_packer.set(buffer.get(), i, initial_state[i].get_value());
        }
        StateID id = insert_buffered_state(buffer.get());
        cached_initial_state = make_unique<State>(lookup_state(id));
    }
    return *cached_initial_state;
}

const TaskProxy &StateRegistry::get_task_proxy() const {
    return task_proxy;
}

int StateRegistry::get_num_variables() const {
    return num_variables;
}

int StateRegistry::get_bins_per_state() const {
    return state_packer.get_num_bins();
}

static int get_bins_per_state(int_packer::IntPacker state_packer){
    return state_packer.get_num_bins();
}

const int_packer::IntPacker &StateRegistry::get_state_packer() const {
    return state_packer;
}

int StateRegistry::get_state_size_in_bytes() const {
    return get_bins_per_state() * sizeof(PackedStateBin);
}

// static class StateRegistryCategoryPlugin : public plugins::TypedCategoryPlugin<StateRegistry> {
//     public:
//         EvaluatorCategoryPlugin() : TypedCategoryPlugin("StateRegistry") {
//             allow_variable_binding();
//         }
//     }
//     _category_plugin;
    