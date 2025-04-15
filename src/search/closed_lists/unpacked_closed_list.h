//
// Created by workbox on 2025-04-10.
//

#ifndef UNPACKED_CLOSED_LIST_H
#define UNPACKED_CLOSED_LIST_H

#include "../utils/treedbs.h"
#include "../closed_list.h"

namespace unpacked_closed_list {

class UnpackedClosedList : public ClosedList {
            TaskProxy task_proxy;
            int num_variables;
            size_t size = 0;


            struct StateIDSemanticHash {
                int variable_count;
                StateIDSemanticHash(
                    int variable_count)
                    : variable_count(variable_count) {
                }

                uint64_t operator()(std::vector<int> data) const {
                    utils::HashState hash_state;
                    for (int i = 0; i < variable_count; ++i) {
                        hash_state.feed(data[i]);
                    }
                    return hash_state.get_hash64();
                }
            };

            struct StateIDSemanticEqual {
                int variable_count;
                StateIDSemanticEqual(int variable_count)
                    : variable_count(variable_count) {
                }

                bool operator()(std::vector<int> lhs, std::vector<int> rhs) const {
                    return std::equal(lhs.begin(), lhs.end(), rhs.begin());
                }
            };

            StateID insert_id_or_pop_state();
        protected:
            using StateIDSet = phmap::flat_hash_set<std::vector<int>, StateIDSemanticHash, StateIDSemanticEqual>;
            StateIDSet registered_states;

            int get_bins_per_state() const;
        public:
            UnpackedClosedList();
            void add_state(const State &state) override;
            bool contains_state(const State &state) override;
            void print() const override;
        };

} // unpacked_closed_list
#endif //UNPACKED_CLOSED_LIST_H
