#include "open_list_factory.h"

#include "plugins/options.h"
#include "plugins/plugin.h"

using namespace std;


template<>
unique_ptr<StateOpenList> OpenListFactory::create_open_list() {
    return create_state_open_list();
}

template<>
unique_ptr<EdgeOpenList> OpenListFactory::create_open_list() {
    return create_edge_open_list();
}

void add_open_list_options_to_feature(
    plugins::Feature &feature) {
    feature.add_option<bool>(
        "pref_only",
        "insert only nodes generated by preferred operators",
        "false");
}

tuple<bool> get_open_list_arguments_from_options(
    const plugins::Options &opts) {
    return make_tuple(opts.get<bool>("pref_only"));
}

static class OpenListFactoryCategoryPlugin : public plugins::TypedCategoryPlugin<OpenListFactory> {
public:
    OpenListFactoryCategoryPlugin() : TypedCategoryPlugin("OpenList") {
        // TODO: use document_synopsis() for the wiki page.
    }
}
_category_plugin;
