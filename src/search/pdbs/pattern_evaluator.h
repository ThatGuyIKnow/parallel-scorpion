#ifndef PDBS_PATTERN_EVALUATOR_H
#define PDBS_PATTERN_EVALUATOR_H

#include "types.h"

#include "../task_proxy.h"

#include "../algorithms/array_pool.h"

#include <vector>

namespace priority_queues {
template<typename Value>
class AdaptiveQueue;
}

namespace pdbs {
class MatchTree;

enum class DeadEndTreatment {
    IGNORE,
    ALL,
    NEW,
};

struct AbstractBackwardOperator {
    int concrete_operator_id;
    int hash_effect;

    AbstractBackwardOperator(
        int concrete_operator_id,
        int hash_effect)
        : concrete_operator_id(concrete_operator_id),
          hash_effect(hash_effect) {
    }
};

struct OperatorInfo {
    int concrete_operator_id;
    std::vector<FactPair> preconditions;
    std::vector<FactPair> effects;
public:
    explicit OperatorInfo(const OperatorProxy &op);
};

struct TaskInfo {
    int num_variables;
    std::vector<OperatorInfo> operator_infos;

    /* The bit at position op_id * num_variables + var is true iff the operator
       has an effect on variable var. */
    std::vector<bool> variable_effects;
public:
    explicit TaskInfo(const TaskProxy &task_proxy);

    bool operator_affects_pattern(const Pattern &pattern, int op_id) const;
};

class PartialStateCollection {
    array_pool::ArrayPool<FactPair> partial_states;
public:
    void add(std::vector<FactPair> &&facts);
    bool subsumes(const std::vector<FactPair> &facts) const;
};

using AbstractOperatorSet = utils::HashMap<
    std::vector<FactPair>, PartialStateCollection>;

class PatternEvaluator {
    TaskProxy task_proxy;

    std::vector<AbstractBackwardOperator> abstract_backward_operators;
    std::unique_ptr<pdbs::MatchTree> match_tree_backward;

    int num_states;

    std::vector<int> goal_states;

    std::vector<int> compute_goal_states(
        const std::vector<std::size_t> &hash_multipliers,
        const std::vector<int> &pattern_domain_sizes,
        const std::vector<int> &variable_to_pattern_index) const;

    void multiply_out(
        const Pattern &pattern,
        const std::vector<size_t> &hash_multipliers,
        int pos,
        int conc_op_id,
        std::vector<FactPair> &prev_pairs,
        std::vector<FactPair> &pre_pairs,
        std::vector<FactPair> &eff_pairs,
        const std::vector<FactPair> &effects_without_pre,
        const std::vector<int> &domain_sizes);

    void build_abstract_operators(
        const Pattern &pattern,
        const std::vector<std::size_t> &hash_multipliers,
        const OperatorInfo &op,
        const std::vector<int> &variable_to_pattern_index,
        const std::vector<int> &domain_sizes);

    /*
      Return true iff all abstract facts hold in the given state.
    */
    bool is_consistent(
        const std::vector<size_t> &hash_multipliers,
        const std::vector<int> &pattern_domain_sizes,
        std::size_t state_index,
        const std::vector<FactPair> &abstract_facts) const;

    bool detects_new_dead_ends(
        const Pattern &pattern,
        const std::vector<int> &distances,
        PartialStateCollection &dead_ends) const;

public:
    PatternEvaluator(
        const TaskProxy &task_proxy,
        const TaskInfo &task_info,
        const pdbs::Pattern &pattern,
        const std::vector<int> &costs);
    ~PatternEvaluator();

    bool is_useful(
        const Pattern &pattern,
        priority_queues::AdaptiveQueue<size_t> &pq,
        PartialStateCollection &dead_ends,
        DeadEndTreatment dead_end_treatment,
        const std::vector<int> &costs) const;
};
}

#endif
