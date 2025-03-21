#include "astar_search.h"

#include "../task_utils/successor_generator.h"

#include "../evaluator.h"
#include "../search_algorithm.h"
#include "../open_list_factory.h"
#include "../option_parser.h"
#include "../plugins/plugin.h"
#include "search_common.h"
#include "../task_utils/task_properties.h"

#include "../utils/logging.h"

#include <algorithm>
#include <cassert>
#include <optional.hh>

using namespace std;

namespace astar_search {


AstarSearch::AstarSearch(
    const shared_ptr<Evaluator> &eval,
    const shared_ptr<ClosedList> &closed,
    OperatorCost cost_type,
    int bound,
    double max_time,
    const string &description,
    utils::Verbosity verbosity)
    : SearchAlgorithm(cost_type, bound, max_time, description, verbosity),
      iteration(0),
      closed_list(closed),
      open_node_registry(OpenAstarNode::no_node),
      open_state_registry(task_proxy) {
    create_open_list_and_f_eval(eval, verbosity);
    if (eval->does_cache_estimates()) {
        cerr << "Error: set cache_estimates=false for heuristics with this A* version." << endl;
        utils::exit_with(utils::ExitCode::SEARCH_INPUT_ERROR);
    }
}

void AstarSearch::initialize() {
    cout << "Conducting A* search, (real) bound = " << bound << endl;
    State initial_state = open_state_registry.get_initial_state();
    open_node_registry[initial_state] = OpenAstarNode(0, nullptr, OperatorID::no_operator);
    EvaluationContext eval_context(initial_state, 0, true, &statistics); 

    statistics.inc_evaluated_states();

    if (open_list->is_dead_end(eval_context)) {
        log << "Initial state is a dead end." << endl;
    } else {
        if (search_progress.check_progress(eval_context))
            statistics.print_checkpoint_line(0);
         start_f_value_statistics(eval_context);

         open_list->insert(eval_context, initial_state.get_id());
    }  
    
    print_initial_evaluator_values(eval_context);
}

void AstarSearch::start_f_value_statistics(EvaluationContext &eval_context) {
    if (f_evaluator) {
        int f_value = eval_context.get_evaluator_value(f_evaluator.get());
        statistics.report_f_value_progress(f_value);
    }
}

void AstarSearch::update_f_value_statistics(EvaluationContext &eval_context) {
    if (f_evaluator) {
        int f_value = eval_context.get_evaluator_value(f_evaluator.get());
        statistics.report_f_value_progress(f_value);
    }
}

void AstarSearch::create_open_list_and_f_eval(const shared_ptr<Evaluator> &eval, utils::Verbosity verbosity) {
    using GEval = g_evaluator::GEvaluator;
    using SumEval = sum_evaluator::SumEvaluator;

    shared_ptr<GEval> g = plugins::make_shared_from_arg_tuples<GEval>("verbosity", verbosity);


    f_evaluator = plugins::make_shared_from_arg_tuples<SumEval>(
        vector<shared_ptr<Evaluator>>{g, eval},
        "", utils::Verbosity(verbosity)
    );
    vector<shared_ptr<Evaluator>> evals = {f_evaluator, eval};

    open_list = plugins::make_unique_from_arg_tuples<AstarOpenList>(evals);
}

void AstarSearch::print_statistics() const {
    closed_list->print();
    statistics.print_detailed_statistics();
    open_state_registry.print_statistics(log);
}

SearchStatus AstarSearch::step() {
    if (open_list->empty()) {
        log << "Completely explored state space -- no solution!" << endl;
        return FAILED;
    }
    auto res = open_list->remove_min_with_f_val();
    StateID id = res.first;
    State s = open_state_registry.lookup_state(id);
    while(true) {
        const OpenAstarNode &node = open_node_registry[s];
        if(node != OpenAstarNode::no_node){
            /*
            As OpenStateRegistry reuses ids, we here verify that the state corresponding to the id 
            stored by the open list has the same f-value as the state that was added into the open list
            */
            EvaluationContext eval_context(s, node.g, false, &statistics);
            int f_val = eval_context.get_evaluator_value(f_evaluator.get());
            if(f_val == res.second) {
                closed_list->add_state(s);
                statistics.inc_expanded();
                statistics.report_f_value_progress(f_val);
                break;            
            }
        }
        if (open_list->empty()) {
            log << "Completely explored state space -- no solution!" << endl;
            return FAILED;
        }
        res = open_list->remove_min_with_f_val();
        id = res.first;
        s = open_state_registry.lookup_state(id);
    }
    const OpenAstarNode &node = open_node_registry[s];  

    if (task_properties::is_goal_state(task_proxy, s)){
        Plan operator_sequence = { node.creating_operator };
        
        for (shared_ptr<ClosedAstarNode> n = node.parent; n->parent != nullptr; n = n->parent)
        {
            operator_sequence.push_back(n->creating_operator);
        }
        reverse(operator_sequence.begin(), operator_sequence.end());
        plan_manager.save_plan(operator_sequence, task_proxy);
        set_plan(operator_sequence);
        return SOLVED;
    }

    vector<OperatorID> applicable_ops;
    successor_generator.generate_applicable_ops(s, applicable_ops);
    OperatorsProxy operators = task_proxy.get_operators();

    std::shared_ptr<ClosedAstarNode> parent = std::make_shared<ClosedAstarNode>(node);
    
    for (OperatorID op_id : applicable_ops) {
        OperatorProxy op = operators[op_id];
        if ((node.g + op.get_cost()) >= bound)
            continue;
        State succ_state = s.get_unregistered_successor(op);        
        statistics.inc_generated();

        if (closed_list->contains_state(succ_state)){
            continue;
        }

        open_state_registry.register_state(succ_state);
        OpenAstarNode &succ_node = open_node_registry[succ_state];

        int succ_g = node.g + get_adjusted_cost(op);

        if (succ_node == OpenAstarNode::no_node) {
            // We have not seen this state before.
            // Evaluate and create a new node.

            EvaluationContext succ_eval_context(
                succ_state, succ_g, false, &statistics);
            statistics.inc_evaluated_states();

            if (open_list->is_dead_end(succ_eval_context)) {
                closed_list->add_state(succ_state);
                succ_node = OpenAstarNode::no_node;
                open_state_registry.unregister_state(succ_state);
                statistics.inc_dead_ends();
                continue;
            }

            succ_node = OpenAstarNode(succ_g, parent, op_id);

            open_list->insert(succ_eval_context, succ_state.get_id());
            if (search_progress.check_progress(succ_eval_context)) {
                 statistics.print_checkpoint_line(succ_g);
            }
        } else if (succ_node.g > succ_g) {
            // We found a new cheapest path to an open state.

            succ_node = OpenAstarNode(succ_g, parent, op_id);

            EvaluationContext succ_eval_context(
                succ_state, succ_node.g, false, &statistics);

            open_list->insert(succ_eval_context, succ_state.get_id()); 
        }
    }
    open_node_registry[s] = OpenAstarNode::no_node;
    open_state_registry.unregister_state(s);
    return IN_PROGRESS;
}

void AstarSearch::save_plan_if_necessary() {
    // We don't need to save here, as we automatically save plans when we find them.
}


class AstarmodSearchFeature
        : public plugins::TypedFeature<SearchAlgorithm, AstarSearch> {
public:
    AstarmodSearchFeature() : TypedFeature("astarmod") {
        document_title(
                "A* search");
        document_synopsis(
                "A* search with open and closed states stored separately.");
        add_option<shared_ptr<Evaluator>>(
            "eval",
            "evaluator for h-value. Make sure to use cache_estimates=false.");

        add_option<shared_ptr<ClosedList>>(
            "closed",
            "list used to store closed nodes.",
            "loes");

        utils::add_log_options_to_feature(*this);
    }

    virtual shared_ptr<AstarSearch> create_component(
        const plugins::Options &options) const override {
        return plugins::make_shared_from_arg_tuples<AstarSearch>(
            options.get<shared_ptr<Evaluator>>("eval"),
            options.get<shared_ptr<ClosedList>>("closed"),
            options.get<OperatorCost>("cost_type"),
            options.get<int>("bound"),
            options.get<double>("max_time"),
            options.get<string>("description"),
            utils::get_log_arguments_from_options(options)
        );
    }
};

static plugins::FeaturePlugin<AstarmodSearchFeature> _plugin;
}