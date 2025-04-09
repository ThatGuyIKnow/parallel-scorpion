#ifndef CLOSED_LIST_H
#define CLOSED_LIST_H

#include "task_proxy.h"
#include "options/option_parser.h"
#include "utils/logging.h"


class ClosedList {
public:
    virtual ~ClosedList() {}
    virtual void add_state(const State &state) = 0;
    virtual bool contains_state(const State &state) = 0;
    virtual void print() const = 0;

    static void add_options_to_parser(plugins::Feature &feature);
};
extern void add_closed_list_options_to_parser(plugins::Feature &feature);

#endif