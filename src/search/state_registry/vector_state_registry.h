#ifndef VECTOR_STATE_REGISTRY_H
#define VECTOR_STATE_REGISTRY_H

#include "state_registry.h"

#include "../abstract_task.h"
#include "../axioms.h"
#include "../state_id.h"

#include "../algorithms/int_hash_set.h"
#include "../algorithms/int_packer.h"
#include "../algorithms/segmented_vector.h"
#include "../algorithms/subscriber.h"
#include "../utils/hash.h"

#include <set>

namespace int_packer {
class IntPacker;
}

using PackedStateBin = int_packer::IntPacker::Bin;


class VectorStateRegistry : public StateRegistry {
    struct StateIDSemanticHash {
        const segmented_vector::SegmentedArrayVector<PackedStateBin> &state_data_pool;
        int state_size;
        StateIDSemanticHash(
            const segmented_vector::SegmentedArrayVector<PackedStateBin> &state_data_pool,
            int state_size)
            : state_data_pool(state_data_pool),
              state_size(state_size) {
        }

        int_hash_set::HashType operator()(int id) const {
            const PackedStateBin *data = state_data_pool[id];
            utils::HashState hash_state;
            for (int i = 0; i < state_size; ++i) {
                hash_state.feed(data[i]);
            }
            return hash_state.get_hash32();
        }
    };

    struct StateIDSemanticEqual {
        const segmented_vector::SegmentedArrayVector<PackedStateBin> &state_data_pool;
        int state_size;
        StateIDSemanticEqual(
            const segmented_vector::SegmentedArrayVector<PackedStateBin> &state_data_pool,
            int state_size)
            : state_data_pool(state_data_pool),
              state_size(state_size) {
        }

        bool operator()(int lhs, int rhs) const {
            const PackedStateBin *lhs_data = state_data_pool[lhs];
            const PackedStateBin *rhs_data = state_data_pool[rhs];
            return std::equal(lhs_data, lhs_data + state_size, rhs_data);
        }
    };

    /*
      Hash set of StateIDs used to detect states that are already registered in
      this registry and find their IDs. States are compared/hashed semantically,
      i.e. the actual state data is compared, not the memory location.
    */
    using StateIDSet = int_hash_set::IntHashSet<StateIDSemanticHash, StateIDSemanticEqual>;
    
    segmented_vector::SegmentedArrayVector<PackedStateBin> state_data_pool;
    StateIDSet registered_states;

    StateID insert_id_or_pop_state();
public:
    explicit VectorStateRegistry(const TaskProxy &task_proxy);
    


    StateID insert_buffered_state(const PackedStateBin *buffer) override;
    State lookup_state(StateID id) const override;
    State lookup_state(StateID id, std::vector<int> &&state_values) const override;
    State get_successor_state(const State &predecessor, const OperatorProxy &op) override;
    size_t size() const override;
    const State &get_initial_state() override;
    void print_statistics(utils::LogProxy &log) const override;

};

#endif