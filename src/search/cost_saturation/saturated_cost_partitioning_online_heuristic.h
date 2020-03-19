#ifndef COST_SATURATION_SATURATED_COST_PARTITIONING_ONLINE_HEURISTIC_H
#define COST_SATURATION_SATURATED_COST_PARTITIONING_ONLINE_HEURISTIC_H

#include "types.h"
#include "unsolvability_heuristic.h"

#include "../heuristic.h"

#include <memory>
#include <vector>

namespace utils {
class Timer;
}

namespace cost_saturation {
class OrderGenerator;

class SaturatedCostPartitioningOnlineHeuristic : public Heuristic {
    const std::shared_ptr<OrderGenerator> cp_generator;
    const Abstractions abstractions;
    CPHeuristics cp_heuristics;
    UnsolvabilityHeuristic unsolvability_heuristic;
    const int interval;
    const bool skip_seen_orders;
    const double max_time;
    const bool diversify;

    const std::vector<int> costs;

    std::vector<int> fact_id_offsets;
    std::vector<bool> seen_facts;
    std::vector<std::vector<bool>> seen_fact_pairs;

    utils::HashSet<Order> seen_orders;
    std::unique_ptr<utils::Timer> timer;
    int num_duplicate_orders;
    int num_evaluated_states;
    int num_scps_computed;

    // For statistics.
    mutable std::vector<int> num_best_order;

    void print_statistics() const;
    int get_fact_id(int var, int value) const;
    bool should_compute_scp(const State &state);

protected:
    virtual int compute_heuristic(const GlobalState &state) override;

public:
    SaturatedCostPartitioningOnlineHeuristic(
        const options::Options &opts,
        Abstractions &&abstractions,
        CPHeuristics &&cp_heuristics,
        UnsolvabilityHeuristic &&unsolvability_heuristic);
    virtual ~SaturatedCostPartitioningOnlineHeuristic() override;
};
}

#endif
