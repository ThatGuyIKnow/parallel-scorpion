#ifndef PACKED_TREEDBS_CLOSED_LIST_H
#define PACKED_TREEDBS_CLOSED_LIST_H

#include "../utils/treedbs.h"
#include "../closed_list.h"


using StatePacker = int_packer::IntPacker;
using PackedStateBin = StatePacker::Bin;
namespace packed_treedbs_closed_list {

    class PackedTreeDBSClosedList  : public ClosedList  {
        std::unique_ptr<utils::TreeDBS> dbs;
        TaskProxy task_proxy;
        const StatePacker &state_packer;
        size_t size = 0;


    public:
        PackedTreeDBSClosedList();
        void add_state(const State &state) override;
        bool contains_state(const State &state) override;
        void print() const override;
    };

} // treedbs_closed_list

#endif //TREEDBS_CLOSED_LIST_H
