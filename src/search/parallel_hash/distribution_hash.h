/*
 * ZobristHash.h
 *
 *  Created on: Oct 18, 2015
 *      Author: yuu
 */

#ifndef DISTRIBUTIONHASH_H
#define DISTRIBUTIONHASH_H
#include "../heuristics/domain_transition_graph.h"
#include "../state_registry.h"

#include "../task_proxy.h"

#include <vector>

namespace domain_transition_graph {
    class DomainTransitionGraph;
}

using std::vector;

namespace distribution_hash {
    using PackedStateBin = int_packer::IntPacker::Bin;
    

    // TODO: Maybe we can group up map based hashes (zbr, abst, sz,...)
    class MapBasedHash {
        std::vector<std::unique_ptr<domain_transition_graph::DomainTransitionGraph>> transition_graphs;
        void get_sucessor_dtg(domain_transition_graph::DomainTransitionGraph* dtg, 
            unsigned int value, vector<int> &result);
    public:
        MapBasedHash(bool is_polynomial);
        unsigned int hash(const State& state);
        unsigned int hash(const std::shared_ptr<State> state);
        unsigned int hash_incremental(const State& predecessor,
            const unsigned int parent_d_hash, const OperatorProxy &op);
    protected:
        const std::shared_ptr<AbstractTask> task;
        TaskProxy task_proxy;
        StateRegistry state_registry;
        std::vector<int> get_frequency_rank();
        bool is_polynomial;

        std::vector<int> reverse_iter_to_val(std::vector<int> in);
        void divideIntoTwo(unsigned int var,
                std::vector<std::vector<unsigned int> >& structures);
        std::vector<std::vector<unsigned int> > map;

    };

    class ZobristHash: public MapBasedHash {
    public:
        ZobristHash(bool is_polynomial);
        static constexpr const char* hash_name() { return "zobrist"; };
    };

    // Well, let's call it AbstractionHash as Abstraction is way too overloaded.
    class AbstractionHash: public MapBasedHash {
    public:
        AbstractionHash(const int abstraction, bool is_polynomial);
        static constexpr const char* hash_name(){return "abstraction"; };

    private:
        int abstraction;
    };

    // Well, let's call it AbstractionHash as Abstraction is way too overloaded.
    class AdaptiveAbstractionHash: public MapBasedHash {
    public:
        explicit AdaptiveAbstractionHash(const double abstraction_ratio, bool is_polynomial);
        static constexpr const char* hash_name() {return "aabstraction"; };

    private:
        double abstraction_ratio;
    };

    class FeatureBasedStructuredZobristHash: public MapBasedHash {
    public:
        FeatureBasedStructuredZobristHash(const int abstraction, bool is_polynomial);
        static constexpr const char* hash_name() { return "fstructured"; };

    private:
        int abstraction;

    };

    class ActionBasedStructuredZobristHash: public MapBasedHash {
    public:
        ActionBasedStructuredZobristHash(const int abstraction, bool is_polynomial);
        static constexpr const char* hash_name() { return "astructured";} ;
    private:
        int abstraction;
    };
}
#endif /* ZOBRISTHASH_H_ */
