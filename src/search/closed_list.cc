#include "closed_list.h"

#include "pdbs/pattern_collection_generator_systematic.h"
#include "plugins/plugin.h"

#include "utils/system.h"

void add_closed_list_options_to_parser(plugins::Feature &feature) {
    utils::add_log_options_to_feature(feature);
}

// static PluginTypePlugin<ClosedList> _type_plugin(
//     "Closed List",
//     "");

// void ClosedList::add_options_to_parser(OptionParser &parser) {
//     add_closed_list_options_to_parser(parser);
// }

// class ClosedListFeature : public plugins::TypedFeature<>


void ClosedList::add_options_to_parser(plugins::Feature &feature) {
    add_closed_list_options_to_parser(feature);
}

static class ClosedListCategoryPlugin : public plugins::TypedCategoryPlugin<ClosedList> {
    public:
        ClosedListCategoryPlugin() : TypedCategoryPlugin("ClosedList") {
            document_synopsis("Type of closed list.");
        }
}
_category_plugin;
