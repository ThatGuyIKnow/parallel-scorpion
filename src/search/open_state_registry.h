#ifndef OPEN_STATE_REGISTRY_H
#define OPEN_STATE_REGISTRY_H

#include "state_registry.h"

#include <unordered_set>
#include <stack>


class OpenStateRegistry : public StateRegistry {
    using UnusedStateIDSet = std::stack<int>;
    UnusedStateIDSet unused_ids;
    const std::unique_ptr<PackedStateBin[]> reusable_buffer;

    StateID insert_id_or_pop_state() override;
    std::pair<StateID, bool> try_insert_unused_id();
public:
    explicit OpenStateRegistry(const TaskProxy &task_proxy);

    void unregister_state(State &state);

    void register_state(State &state);

    /*
      Returns the size of the state_data_pool.
    */
    size_t size() const override{
        return state_data_pool.size();
    }

    /*
      Returns the number of states registered.
    */
    size_t registered_size() const {
        return registered_states.size();
    }

    void print_statistics(utils::LogProxy &log) const override;    
};

#endif
