#include "astar_search.h"
#include "eager_search.h"
#include "search_common.h"

#include "../plugins/plugin.h"


using namespace std;

namespace plugin_astar {
class AStarSearchFeature
    : public plugins::TypedFeature<SearchEngine, eager_search::EagerSearch> {
public:
    AStarSearchFeature() : TypedFeature("astar") {
        document_title("A* search (eager)");
        document_synopsis(
            "A* is a special case of eager best first search that uses g+h "
            "as f-function. "
            "We break ties using the evaluator. Closed nodes are re-opened.");

        add_option<shared_ptr<Evaluator>>("eval", "evaluator for h-value");
        add_option<shared_ptr<Evaluator>>(
            "lazy_evaluator",
            "An evaluator that re-evaluates a state before it is expanded.",
            plugins::ArgumentInfo::NO_DEFAULT);
        astar_search::AstarSearch::add_option_to_parser(
            *this, "astar");
        eager_search::add_options_to_parser(*this);

        document_note(
            "lazy_evaluator",
            "When a state s is taken out of the open list, the lazy evaluator h "
            "re-evaluates s. If h(s) changes (for example because h is path-dependent), "
            "s is not expanded, but instead reinserted into the open list. "
            "This option is currently only present for the A* algorithm.");
        document_note(
            "Equivalent statements using general eager search",
            "\n```\n--search astar(evaluator)\n```\n"
            "is equivalent to\n"
            "```\n--evaluator h=evaluator\n"
            "--search eager(tiebreaking([sum([g(), h]), h], unsafe_pruning=false),\n"
            "               reopen_closed=true, f_eval=sum([g(), h]))\n"
            "```\n", true);
    }

    virtual shared_ptr<eager_search::EagerSearch>
    create_component(const plugins::Options &opts) const override {



        shared_ptr<eager_search::EagerSearch> engine;
        auto temp = search_common::create_astar_open_list_factory_and_f_eval(opts.get<shared_ptr<Evaluator>>("eval"), opts.get<utils::Verbosity>("verbosity"));
        vector<shared_ptr<Evaluator>> preferred_list;
        engine = plugins::make_shared_from_arg_tuples<eager_search::EagerSearch>(temp.first, temp.second, true, preferred_list);
        return engine;
    }
};

static plugins::FeaturePlugin<AStarSearchFeature> _plugin;
}

