#include "parallel_eager_search.h"
#include "eager_search.h"
#include "search_common.h"
#include "../parallel_hash/distribution_hash.h"

#include "../plugins/plugin.h"

using namespace std;
namespace plugin_parallel_eager {

class ParallelEagerSearchFeature
    : public plugins::TypedFeature<SearchAlgorithm, parallel_eager_search::ParallelEagerSearch> {
public:
    ParallelEagerSearchFeature() : TypedFeature("peager") {
        document_title("Parallel eager best-first search");
        document_synopsis("Parallelized version of eager best-first search, using disjoint memory and MPI.");

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
        add_option<shared_ptr<distribution_hash::MapBasedHash>>(
            "hash",
            "distributed hashing function",
            plugins::ArgumentInfo::NO_DEFAULT
        );
        eager_search::add_eager_search_options_to_feature(
            *this, "peager");
    }

    virtual shared_ptr<parallel_eager_search::ParallelEagerSearch> create_component(
        const plugins::Options &opts,
        const utils::Context &) const override {
        return plugins::make_shared_from_arg_tuples<parallel_eager_search::ParallelEagerSearch>(
            opts.get<shared_ptr<OpenListFactory>>("open"),
            opts.get<bool>("reopen_closed"),
            opts.get<shared_ptr<Evaluator>>("f_eval", nullptr),
            opts.get_list<shared_ptr<Evaluator>>("preferred"),
            opts.get<shared_ptr<distribution_hash::MapBasedHash>>("hash"),
            eager_search::get_eager_search_arguments_from_options(opts)
            );
    }
};

static plugins::FeaturePlugin<ParallelEagerSearchFeature> _plugin;
}
