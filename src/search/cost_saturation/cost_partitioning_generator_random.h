#ifndef COST_SATURATION_COST_PARTITIONING_GENERATOR_RANDOM_H
#define COST_SATURATION_COST_PARTITIONING_GENERATOR_RANDOM_H

#include "cost_partitioning_generator.h"

namespace utils {
class RandomNumberGenerator;
}

namespace cost_saturation {
class Abstraction;

class CostPartitioningGeneratorRandom : public CostPartitioningGenerator {
    const std::shared_ptr<utils::RandomNumberGenerator> rng;
    std::vector<int> order;

protected:
    virtual void initialize(
        const TaskProxy &task_proxy,
        const std::vector<std::unique_ptr<Abstraction>> &abstractions,
        const std::vector<int> &costs);

public:
    explicit CostPartitioningGeneratorRandom(const options::Options &opts);

    virtual CostPartitioning get_next_cost_partitioning(
        const TaskProxy &task_proxy,
        const std::vector<std::unique_ptr<Abstraction>> &abstractions,
        const std::vector<int> &costs,
        CPFunction cp_function) override;
};
}

#endif
