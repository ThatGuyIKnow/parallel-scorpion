#include "open_state_registry.h"

#include "per_state_information.h"
#include "task_proxy.h"

#include "task_utils/task_properties.h"
#include "utils/logging.h"

using namespace std;

OpenStateRegistry::OpenStateRegistry(const TaskProxy &task_proxy)
    : StateRegistry(task_proxy),
    reusable_buffer(std::make_unique<PackedStateBin[]>(get_bins_per_state())) {
        fill_n(reusable_buffer.get(), get_bins_per_state(), 0);
}

StateID OpenStateRegistry::insert_id_or_pop_state()
{
    int last_id = state_data_pool.size() - 1;
    if (!unused_ids.empty())
    {
        int id = unused_ids.top();
        *state_data_pool[id] = *state_data_pool[last_id];
        state_data_pool.pop_back();
        pair<StateID, bool> result = try_insert_unused_id();
        
        return result.first;
    }

    StateID id(last_id);
    pair<int, bool> result = registered_states.insert(id.value);
    bool is_new_entry = result.second;
    if (!is_new_entry) {
        state_data_pool.pop_back();
    }
    assert(registered_states.size() + unused_ids.size() == state_data_pool.size());
    return StateID(result.first);
}

std::pair<StateID, bool> OpenStateRegistry::try_insert_unused_id()
{
    int id = unused_ids.top();

    pair<int, bool> result = registered_states.insert(id);
    bool is_new_entry = result.second;
    
    if(is_new_entry){
        unused_ids.pop();
    }
    assert(registered_states.size() + unused_ids.size() == state_data_pool.size());
    return std::pair<StateID, bool>(StateID(result.first), is_new_entry);
}


void OpenStateRegistry::unregister_state(State &state)
{
    assert(state.get_registry() == this);
    StateID id = state.get_id();
    registered_states.erase(id.value);
    unused_ids.push(id.value);
    state.registry = nullptr;
    state.buffer = nullptr;
    state.id = StateID::no_state;
}

void OpenStateRegistry::register_state(State &state)
{
    PackedStateBin *buffer;
    bool reuse_id = false;
    if(unused_ids.empty()){
        state_data_pool.push_back(reusable_buffer.get());
        buffer = state_data_pool[state_data_pool.size() - 1];
    } else {
        buffer = state_data_pool[unused_ids.top()];
        reuse_id = true;
    }
    for (size_t i = 0; i < state.size(); ++i) {
        state_packer.set(buffer, i, state[i].get_value());
    }
    if(reuse_id){
        auto res = try_insert_unused_id();
        StateID id = res.first;
        state = lookup_state(id);
    } else {
        StateID id = insert_id_or_pop_state();
        state = lookup_state(id);
    }
}

void OpenStateRegistry::print_statistics(utils::LogProxy &log) const {
    log << "Number of registered states: " << registered_size() << endl;
    log << "Max number of registered states: " << size() << endl;
    registered_states.print_statistics(log);
}
