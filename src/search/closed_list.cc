#include "closed_list.h"

#include "option_parser.h"
#include "plugin.h"

#include "utils/system.h"

void add_closed_list_options_to_parser(options::OptionParser &parser) {
    utils::add_log_options_to_parser(parser);
}

static PluginTypePlugin<ClosedList> _type_plugin(
    "Closed List",
    "");

void ClosedList::add_options_to_parser(OptionParser &parser) {
    add_closed_list_options_to_parser(parser);
}