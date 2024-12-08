#include "parallel_eager_search.h"
#include "search_common.h"

#include "../plugins/plugin.h"

using namespace std;

namespace plugin_parallel_eager {
class ParallelEagerSearchFeature : public plugins::TypedFeature<SearchAlgorithm, parallel_eager_search::ParallelEagerSearch> {
public:
    ParallelEagerSearchFeature() : TypedFeature("peager") {
        document_title("Parallel eager best-first search");
        document_synopsis("");

        add_option<shared_ptr<OpenListFactory>>("open", "open list");
        add_option<bool>(
            "reopen_closed",
            "reopen closed nodes",
            "false");
        add_option<shared_ptr<Evaluator>>(
            "f_eval",
            "set evaluator for jump statistics. "
            "(Optional; if no evaluator is used, jump statistics will not be displayed.)",
            plugins::ArgumentInfo::NO_DEFAULT);
        add_list_option<shared_ptr<Evaluator>>(
            "preferred",
            "use preferred operators of these evaluators",
            "[]");
        parallel_eager_search::add_options_to_feature(*this);
    }
};

static plugins::FeaturePlugin<ParallelEagerSearchFeature> _plugin;
}
