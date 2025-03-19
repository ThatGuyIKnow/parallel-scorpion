#ifndef SEARCH_ENGINES_ASTAR_SEARCH_H
#define SEARCH_ENGINES_ASTAR_SEARCH_H

#include "../search_engine.h"

#include "../closed_list.h"

#include "../evaluation_context.h"
#include "../evaluator.h"
#include "../evaluators/g_evaluator.h"
#include "../evaluators/sum_evaluator.h"
#include "../option_parser_util.h"

#include "../utils/hash.h"
#include "../open_state_registry.h"

#include "../per_state_information.h"

#include <vector>
#include <map>
#include <deque>

using namespace std;

namespace astar_search {
struct ClosedAstarNode;


struct OpenAstarNode {
    static const int INFTY = numeric_limits<int>::max();
    int g;
    shared_ptr<ClosedAstarNode> parent;
    OperatorID creating_operator;

    static const OpenAstarNode no_node;

    OpenAstarNode()
        : g(-1),
          parent(nullptr),
          creating_operator(OperatorID::no_operator) {}

    OpenAstarNode(int g, shared_ptr<ClosedAstarNode> parent, OperatorID creating_operator)
        : g(g),
          parent(parent),
          creating_operator(creating_operator) {
    }

    bool operator==(const OpenAstarNode& other) const{
        return (creating_operator == other.creating_operator) && (g == other.g) && (parent == other.parent);
    }
    bool operator!=(const OpenAstarNode& other) const {
        return !operator==(other);
    }
    bool is_dead_end() {
        return (g == INFTY) && (parent == nullptr);
    }
    void mark_as_dead_end() {
        g = INFTY;
        parent = nullptr;
    }
};
const OpenAstarNode OpenAstarNode::no_node = OpenAstarNode();

struct ClosedAstarNode {
    shared_ptr<ClosedAstarNode> parent;
    OperatorID creating_operator;

    ClosedAstarNode(shared_ptr<ClosedAstarNode> parent, OperatorID creating_operator)
        : parent(parent),
          creating_operator(creating_operator) {
    }

    ClosedAstarNode(OpenAstarNode node)
        : parent(node.parent),
          creating_operator(node.creating_operator) {
    }
};

//Pretty much a copy of TiebreakingOpenList with the exception being that remove_min now also returns f-value
class AstarOpenList {
    using Bucket = deque<StateID>;

    map<const vector<int>, Bucket> buckets;
    int size;

    vector<shared_ptr<Evaluator>> evaluators;

public:
    explicit AstarOpenList(const Options &opts);

    bool empty() const;
    void insert(EvaluationContext &eval_context,
                              const StateID &entry);

    pair<StateID, int> remove_min_with_f_val();
    bool is_dead_end(EvaluationContext &eval_context) const;
};

AstarOpenList::AstarOpenList(const Options &opts) :
      size(0), evaluators(opts.get_list<shared_ptr<Evaluator>>("evals")) {
}

void AstarOpenList::insert(
    EvaluationContext &eval_context, const StateID &entry) {
    vector<int> key;
    key.reserve(evaluators.size());
    for (const shared_ptr<Evaluator> &evaluator : evaluators)
        key.push_back(eval_context.get_evaluator_value_or_infinity(evaluator.get()));
    buckets[key].push_back(entry);
    ++size;
}

pair<StateID, int> AstarOpenList::remove_min_with_f_val() {
    assert(size > 0);
    typename map<const vector<int>, Bucket>::iterator it;
    it = buckets.begin();
    assert(it != buckets.end());
    assert(!it->second.empty());
    --size;
    pair<StateID, int> result(it->second.front(), it->first[0]);
    it->second.pop_front();
    if (it->second.empty())
        buckets.erase(it);
    return result;
}

bool AstarOpenList::empty() const {
    return size == 0;
}
 bool AstarOpenList::is_dead_end(EvaluationContext &eval_context) const {
    for (const shared_ptr<Evaluator> &evaluator : evaluators)
        if (!eval_context.is_evaluator_value_infinite(evaluator.get()))
            return false;
        else if(evaluator->dead_ends_are_reliable())
             return true;
    return true;
 }


class AstarSearch : public SearchEngine {    
    unique_ptr<AstarOpenList> open_list;
    shared_ptr<Evaluator> f_evaluator;
    
    int iteration;

    shared_ptr<ClosedList> closed_list;

    PerStateInformation<OpenAstarNode> open_node_registry;
    OpenStateRegistry open_state_registry;

    void start_f_value_statistics(EvaluationContext &eval_context);
    void update_f_value_statistics(EvaluationContext &eval_context);
    
    void create_open_list_and_f_eval(const Options &opts);
protected:
    virtual void initialize() override;
    virtual SearchStatus step() override;

public:
    explicit AstarSearch(const options::Options &opts);

    void save_plan_if_necessary() override;

    virtual void print_statistics() const override;
};
}

#endif