#ifndef TREEDBS_CLOSED_LIST_H
#define TREEDBS_CLOSED_LIST_H

#include "../utils/treedbs.h"
#include "../closed_list.h"

namespace treedbs_closed_list {

    class TreeDBSClosedList  : public ClosedList  {
        std::shared_ptr<utils::TreeDBS> dbs;
        TaskProxy task_proxy;
        size_t size = 0;
    public:
        TreeDBSClosedList();
        void add_state(const State &state) override;
        bool contains_state(const State &state) const override;
        void print() const override;
    };

} // treedbs_closed_list

#endif //TREEDBS_CLOSED_LIST_H
