#ifndef TREEDBS_CLOSED_LIST_H
#define TREEDBS_CLOSED_LIST_H

#include "../utils/treedbs.h"
#include "../closed_list.h"



using PackedStateBin = int_packer::IntPacker::Bin;
namespace vector_closed_list {

    class VectorClosedList  : public ClosedList  {
        TaskProxy task_proxy;
        size_t size = 0;


        struct StateIDSemanticHash {
            int state_size;
            StateIDSemanticHash(
                int state_size)
                : state_size(state_size) {
            }

            uint64_t operator()(std::vector<PackedStateBin> data) const {
                utils::HashState hash_state;
                for (int i = 0; i < state_size; ++i) {
                    hash_state.feed(data[i]);
                }
                return hash_state.get_hash64();
            }
        };

        struct StateIDSemanticEqual {
            int state_size;
            StateIDSemanticEqual(int state_size)
                : state_size(state_size) {
            }

            bool operator()(std::vector<PackedStateBin> lhs, std::vector<PackedStateBin> rhs) const {
                return std::equal(lhs.begin(), lhs.end(), rhs.begin());
            }
        };

        StateID insert_id_or_pop_state();
    protected:
        using StateIDSet = phmap::flat_hash_set<std::vector<PackedStateBin>, StateIDSemanticHash, StateIDSemanticEqual>;
        const int_packer::IntPacker &state_packer;
        StateIDSet registered_states;

        int get_bins_per_state() const;
    public:
        VectorClosedList();
        void add_state(const State &state) override;
        bool contains_state(const State &state) override;
        void print() const override;
    };

} // treedbs_closed_list

#endif //TREEDBS_CLOSED_LIST_H
