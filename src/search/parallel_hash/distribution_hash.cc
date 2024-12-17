#include "distribution_hash.h"

#include "../heuristics/domain_transition_graph.h"
#include "../task_utils/task_properties.h"
#include "../task_utils/successor_generator.h"
#include "../utils/logging.h"
#include "../task_proxy.h"
#include "../tasks/root_task.h"
#include "../plugins/plugin.h"

#include <iostream>
#include <fstream>
#include <string>
#include <random>   


using namespace std;

template<typename T>
bool compare_pair(const pair<int, T> a, const pair<int, T> b) {
	return a.first < b.first;
}
namespace distribution_hash {
    MapBasedHash::MapBasedHash(bool is_polynomial) :
            task(tasks::g_root_task),
            task_proxy(*task),
            state_registry(task_proxy),
            is_polynomial(is_polynomial) {
        domain_transition_graph::DTGFactory factory(task_proxy, true, [](int, int) {return false;});
        transition_graphs = factory.build_dtgs();

        // Initialize map with 0 filled.
        VariablesProxy vars = task_proxy.get_variables();
        map.resize(vars.size());
        for (size_t i = 0; i < map.size(); ++i) {
            map[i].resize(vars[i].get_domain_size());
            std::fill(map[i].begin(), map[i].end(), 0);
        }
    }

    unsigned int MapBasedHash::hash(const State& state) {
        unsigned int r = 0;
        state.unpack();
        std::vector<int> unpacked_state = state.get_unpacked_values();
        if (!is_polynomial) {
            for (size_t i = 0; i < map.size(); ++i) {
                r = r ^ map[i][unpacked_state[i]];
            }
        } else {
            unsigned int last_size = 1;
            for (size_t i = 0; i < map.size(); ++i) {
                if (map[i][0] != 0) {
                    r *= last_size;
                    r += map[i][unpacked_state[i]];
                    last_size = map[i].size();
                }
            }
        }
        return r;
    }

    // TODO: not sure this is actually saving up time, or just messing up things.
    // we CANNOT precompute inc_hash for each operator as we are not sure
    // whether each effect CHANGES the state or not.
    unsigned int MapBasedHash::hash_incremental(const State& predecessor,
            const unsigned int parent_d_hash, const OperatorProxy &op) {
        unsigned int ret = parent_d_hash;
        
        if (task_properties::is_applicable(op, predecessor)) {
            EffectsProxy post = op.get_effects();
            predecessor.unpack();
            vector<int> predecessor_values = predecessor.get_unpacked_values();

            for (EffectProxy effect : op.get_effects()) {
                if (does_fire(effect, predecessor)) {
                    FactPair effect_pair = effect.get_fact().get_pair();
                    int predecessor_value = predecessor_values[effect_pair.var];
                    int value_flipped = map[effect_pair.var][effect_pair.value]
                            ^ map[effect_pair.var][predecessor_value];
                    ret = ret ^ value_flipped;
                }
            }
        } 

        return ret;
    }

    void MapBasedHash::get_sucessor_dtg(domain_transition_graph::DomainTransitionGraph* dtg, 
        unsigned int value, vector<int> &result) {

        const vector<domain_transition_graph::ValueTransition> &transitions = dtg->nodes[value].transitions;
        result.reserve(transitions.size());
        for (size_t i = 0; i < transitions.size(); i++){
            result.push_back(transitions[i].target->value);
        }

    }

    // TODO: This method is messy. As it is not the core of this program, ill let it for now.
    void MapBasedHash::divideIntoTwo(unsigned int var,
            vector<vector<unsigned int> >& structures) {
        domain_transition_graph::DomainTransitionGraph* dtg = transition_graphs[var].get();
        
        std::vector<unsigned int> structure; // = structure_index
        std::vector<unsigned int> transitions;
        for (size_t p = 0; p < map[var].size(); ++p) {
            vector<int> successors;
            get_sucessor_dtg(dtg, p, successors);
            transitions.insert(transitions.end(), successors.begin(),
                    successors.end());
        }
        int least_connected = 100000;
        int least_connected_node = 0; // in index
        for (size_t p = 0; p < map[var].size(); ++p) {
            int c = std::count(transitions.begin(), transitions.end(), p);
            vector<int> successors;
            get_sucessor_dtg(dtg, p, successors);
            c += successors.size();
            if (c < least_connected) {
                least_connected = c;
                least_connected_node = p;
            }
        }
        structure.push_back(least_connected_node);

        while (structure.size() < map[var].size() / 2) {
            transitions.clear();
            // 2. add a node which is mostly connected to strucuture
            for (size_t p = 0; p < structure.size(); ++p) {
                vector<int> successors;
                get_sucessor_dtg(dtg, p, successors);
                transitions.insert(transitions.end(), successors.begin(),
                        successors.end());
            }

            // count the most connected nodes
            int most_connected = 0;
            int most_connected_node = 0; // in index
            for (size_t p = 0; p < map[var].size(); ++p) {
                // if p is already in the group, then skip that.
                if (find(structure.begin(), structure.end(), p)
                        != structure.end()) {
                    continue;
                }
                int c = std::count(transitions.begin(), transitions.end(), p);
                vector<int> successors;
                get_sucessor_dtg(dtg, p, successors);
                c += successors.size();

                if (c > most_connected) {
                    most_connected = c;
                    most_connected_node = p;
                }
            }

            // if the no nodes connected, then return
            if (most_connected == 0) {
                break;
            }
            structure.push_back(most_connected_node);
        }
        std::sort(structure.begin(), structure.end());
        structures.push_back(structure);

        // the rest of the predicates will be in the second structure.
        // TODO: not sure this assumption is right or not.
        //       maybe not right.
        std::vector<unsigned int> structure2; // = xor_groups[gs] - structure;

        int most_connected = 0;
        int most_connected_node = 0; // in index

        for (size_t p = 0; p < map[var].size(); ++p) {
            if (find(structure.begin(), structure.end(), p) == structure.end()) {
                most_connected_node = p;
                break;
            }
        }

        transitions.clear();
        // 2. add a node which is mostly connected to strucuture
        for (size_t p = 0; p < structure.size(); ++p) {
            vector<int> successors;
            get_sucessor_dtg(dtg, p, successors);
            transitions.insert(transitions.end(), successors.begin(),
                    successors.end());
        }

        for (size_t p = 0; p < map[var].size(); ++p) {
            int c = std::count(transitions.begin(), transitions.end(), p);
            vector<int> successors;
            get_sucessor_dtg(dtg, p, successors);
            c += successors.size();
            if (c > most_connected
                    && find(structure.begin(), structure.end(), p)
                            == structure.end()) {
                most_connected = c;
                most_connected_node = p;
            }
        }
        structure2.push_back(most_connected_node);

        while (true) {
            if (structure.size() + structure2.size() >= map[var].size()) {
                break;
            }
            transitions.clear();
            // 2. add a node which is mostly connected to strucuture
            for (size_t p = 0; p < structure2.size(); ++p) {
                vector<int> successors;
                get_sucessor_dtg(dtg, p, successors);
                transitions.insert(transitions.end(), successors.begin(),
                        successors.end());
            }
            // count the most connected nodes
            int most_connected = 0;
            int most_connected_node = 0; // in index
            for (size_t p = 0; p < map[var].size(); ++p) {
                // if p is already in the group, then skip that.
                if (find(structure2.begin(), structure2.end(), p)
                        != structure2.end()
                        || find(structure.begin(), structure.end(), p)
                                != structure.end()) {
                    continue;
                }
                int c = std::count(transitions.begin(), transitions.end(), p);
                vector<int> successors;
                vector<int> inter;
                get_sucessor_dtg(dtg, p, successors);
                inter.resize(successors.size() + structure2.size());

                vector<int>::iterator it = set_intersection(successors.begin(),
                        successors.end(), structure2.begin(), structure2.end(),
                        inter.begin());
                c += (it - inter.begin());
                if (c > most_connected) {
                    most_connected = c;
                    most_connected_node = p;
                }
            }

            // if the no nodes connected, then return
            if (most_connected == 0) {
                break;
            }
            structure2.push_back(most_connected_node);
        }


        if (structure2.size() > 1) {
            std::sort(structure2.begin(), structure2.end());
            structures.push_back(structure2);
        }
    }

    vector<int> MapBasedHash::get_frequency_rank() {
        // 1.  Count the number of operator which functions the variable.
        vector<pair<int, int> > n_op_functioning;
        VariablesProxy vars = task_proxy.get_variables();
        for (size_t i = 0; i < vars.size(); ++i) {
            n_op_functioning.push_back(pair<int, int>(0, i));
        }

        OperatorsProxy ops = task_proxy.get_operators();
        for (size_t i = 0; i < ops.size(); ++i) {
            const OperatorProxy op = ops[i];
            for (EffectProxy effect : op.get_effects()) {
                FactPair fact = effect.get_fact().get_pair();
                ++n_op_functioning[fact.var].first;
            }
        }

        std::sort(n_op_functioning.begin(), n_op_functioning.end(),
                &compare_pair<int>);

        vector<int> n_op_functioning_rank(vars.size());
        for (size_t i = 0; i < vars.size(); ++i) {
            n_op_functioning_rank[n_op_functioning[i].second] = i;
        }

        return n_op_functioning_rank;
    }

    vector<int> MapBasedHash::reverse_iter_to_val(vector<int> in) {
        vector<int> r;
        r.resize(in.size());
        for (size_t i = 0; i < in.size(); ++i) {
            r[in[i]] = i;
        }
        return r;
    }
    
    ZobristHash::ZobristHash(bool is_polynomial)
        : MapBasedHash(is_polynomial) // Initialize the base class
    {
        std::mt19937 rng(717); 

        std::uniform_int_distribution<uint32_t> dist(0, UINT32_MAX);

        for (size_t i = 0; i < map.size(); ++i) {
            for (size_t j = 0; j < map[i].size(); ++j) {
                map[i][j] = dist(rng); 
            }
        }
    }

    AbstractionHash::AbstractionHash(
        const int abstraction, bool is_polynomial) :
            MapBasedHash(is_polynomial),
            abstraction(abstraction) {


        unsigned int current_size = 1;

        vector<int> n_op_functioning_rank = get_frequency_rank();
        vector<int> rank = reverse_iter_to_val(n_op_functioning_rank);

        std::mt19937 rng(717); 
        std::uniform_int_distribution<uint32_t> dist(0, UINT32_MAX);


        // Abstract out from the most influent variables.

        for (size_t i = 0; i < map.size(); ++i) {
            int k = rank[i];
    //		printf("current_size=%u\n", current_size);
            int current_ratio = current_size * map[k].size();
            if (current_ratio > abstraction) {
                continue;
            }
            current_size *= map[k].size();

            for (size_t j = 0; j < map[k].size(); ++j) {
                map[k][j] = dist(rng);
            }
        }

        for (size_t i = 0; i < map.size(); ++i) {
            for (size_t j = 0; j < map[i].size(); ++j) {
                printf("%u ", map[i][j]);
            }
            printf("\n");
        }

    }

AdaptiveAbstractionHash::AdaptiveAbstractionHash(const double abstraction_ratio, bool is_polynomial)
    : MapBasedHash(is_polynomial),         // Initialize base class
      abstraction_ratio(abstraction_ratio) {
        unsigned int whole_variable_space_size = 1;
        for (size_t i = 0; i < map.size(); ++i) {
            whole_variable_space_size += map[i].size();
        }
        
        unsigned int abstraction_size = uint(whole_variable_space_size
                * (1.0 - abstraction_ratio));
        unsigned int current_size = 0;

        unsigned int abstraction_graph_size = 1;

        vector<int> n_op_functioning_rank = get_frequency_rank();
        vector<int> rank = reverse_iter_to_val(n_op_functioning_rank);

        std::mt19937 rng(717); 
        std::uniform_int_distribution<uint32_t> dist(0, UINT32_MAX);

        // Abstract out from the most influent variables.
        // log << "mapsize=" << map.size() << endl;
        
        // log << "whole_variable_space_size=" << whole_variable_space_size << endl;
        // log << "abstraction_size=" << abstraction_size << endl;

        for (size_t i = 0; i < map.size(); ++i) {
            int k = rank[i];
            if (current_size > abstraction_size) {
                continue;
            }

            current_size += map[k].size();
            abstraction_graph_size *= map[k].size();

            for (size_t j = 0; j < map[k].size(); ++j) {
                map[k][j] = dist(rng);
            }
        }

        // log << "abstraction_graph_size = " <<  abstraction_graph_size << endl;

        // for (size_t i = 0; i < map.size(); ++i) {
        //     for (size_t j = 0; j < map[i].size(); ++j) {
        //         printf("%u ", map[i][j]);
        //     }
        //     printf("\n");
        // }

    }

    FeatureBasedStructuredZobristHash::FeatureBasedStructuredZobristHash(
            const int abstraction, bool is_polynomial) :
            MapBasedHash(is_polynomial),
            abstraction(abstraction) {
        unsigned int whole_variable_space_size = 1;
        for (size_t i = 0; i < map.size(); ++i) {
            whole_variable_space_size += map[i].size();
        }

        unsigned int abstraction_size = whole_variable_space_size * abstraction;
        unsigned int current_size = 1;


        std::mt19937 rng(717); 
        std::uniform_int_distribution<uint32_t> dist(0, UINT32_MAX);

        // TODO: read my last stupid code and implement while i know this is suboptimal.
        for (size_t i = 0; i < map.size(); ++i) {
            if (map[i].size() <= 2) {
                continue;
            }
            current_size += map[i].size();
            if (current_size > abstraction_size) {
                break;
            }
            vector<vector<unsigned int> > structures;
            divideIntoTwo(i, structures);
            for (size_t j = 0; j < structures.size(); ++j) {
                unsigned int r = dist(rng);
                for (size_t k = 0; k < structures[j].size(); ++k) {
                    map[i][structures[j][k]] = r;
                }
            }

        }

        for (size_t i = 0; i < map.size(); ++i) {
            for (size_t j = 0; j < map[i].size(); ++j) {
                if (map[i][j] == 0) {
                    map[i][j] = dist(rng);
                }
            }
        }

        // for (size_t i = 0; i < map.size(); ++i) {
        //     for (size_t j = 0; j < map[i].size(); ++j) {
        //         printf("%u ", map[i][j]);
        //     }
        //     printf("\n");
        // }
    }

    ActionBasedStructuredZobristHash::ActionBasedStructuredZobristHash(
            const int abstraction, bool is_polynomial) :
            MapBasedHash(is_polynomial),
            abstraction(abstraction) {

        std::mt19937 rng(717); 
        std::uniform_int_distribution<uint32_t> dist(0, UINT32_MAX);

        // vector<pair<int, const Operator *> > ops;

        // g_successor_generator->get_op_depths(0, ops);

        // sort(ops.begin(), ops.end(), &compare_pair<const Operator *>);

        // Need to think about seeds later, but for this prototype we put a fixed number.
    
        OperatorsProxy ops = task_proxy.get_operators();

        // First, initialize the map as in ZobristHash. We start it from this values.
        for (size_t i = 0; i < map.size(); ++i) {
            for (size_t j = 0; j < map[i].size(); ++j) {
                map[i][j] = dist(rng);
            }
        }

        // log << "abstraction = " << abstraction << endl;
        unsigned int abstraction_size = ops.size() * abstraction;
        unsigned int current_size = 0;

        // STRUCTURED means that it is used for building structure somewhere.
        // Changing its value will destroy it.
        vector<vector<bool> > has_structured;
        has_structured.resize(map.size());
        for (size_t i = 0; i < map.size(); ++i) {
            has_structured[i].resize(map[i].size());
            fill(has_structured[i].begin(), has_structured[i].end(), false);
        }

        // ad hoc numbers to terminate the algorithm after
        // structuring all possible actions.
        unsigned int failed = 0;
    //	unsigned int max_failed = 60;

        // TODO: KNOWN ISSUE
        // Add effect and delete effect does not always CHANGE the state.
        // Therefore, this method does not GUARANTEE to build a structure.
        // If we could somewhat get around it, we may have a chance to optimize it more.
        size_t op_index = 0;
        while ((current_size < abstraction_size) && (op_index < ops.size())) {
            OperatorProxy op = ops[op_index++];
            vector<pair<int, int> > effects;


            PreconditionsProxy pres = op.get_preconditions();
            EffectsProxy posts = op.get_effects();
            

            for (EffectProxy post : op.get_effects() ) {
                FactPair fact = post.get_fact().get_pair();
                effects.push_back(pair<int, int>(fact.var, fact.value));
            }
            for (FactProxy pre : op.get_preconditions() ) {
                FactPair fact = pre.get_pair();
                effects.push_back(pair<int, int>(fact.var, fact.value));
            }

            // for (size_t i = 0; i < op.get_pre_post().size(); ++i) {
            //     const PrePost &pre_post = op.get_pre_post()[i];
            //     if (pre_post.post >= 0) {
            //         pair<int, int> p(pre_post.var, pre_post.post);
            //         effects.push_back(p);
            //     }
            //     if (pre_post.pre >= 0) {
            //         pair<int, int> p2(pre_post.var, pre_post.pre);
            //         effects.push_back(p2);
            //     }
            // }

            bool succeeded = false;
            // check if ANY of the predicates is NOT structured

            for (size_t i = 0; i < effects.size(); ++i) {

                if (has_structured[effects[i].first][effects[i].second]) {
                    continue;
                } else {
                    unsigned int value = 0;
                    for (size_t j = 0; j < effects.size(); ++j) {
                        if (i != j) {
                            value = value
                                    ^ map[effects[j].first][effects[j].second];
                        }
                    }
                    map[effects[i].first][effects[i].second] = value;
                    succeeded = true;
                    break;
                }
            }
            if (succeeded) {
                for (size_t i = 0; i < effects.size(); ++i) {
                    has_structured[effects[i].first][effects[i].second] = true;
                }
                ++current_size;
                failed = 0;
            } else {
                ++failed;
            }
        }

        // log << "abst succeeded=" << current_size <<  "/" << 
        //    abstraction_size << " : " << ops.size() << endl;

        // for (size_t i = 0; i < map.size(); ++i) {
        //     for (size_t j = 0; j < map[i].size(); ++j) {
        //         printf("%u ", map[i][j]);
        //     }
        //     printf("\n");
        // }
    }

    

    static class DistributionHashCategoryPlugin : public plugins::TypedCategoryPlugin<MapBasedHash> {
    public:
        DistributionHashCategoryPlugin() : TypedCategoryPlugin("DistributionHash") {
            document_synopsis("Distribution Hash used for allocating nodes to ranks.");
            allow_variable_binding();
        }
    }
    _category_plugin;

    class ZobristHashFeatures
        : public plugins::TypedFeature<MapBasedHash, ZobristHash> {
    public:
        ZobristHashFeatures() : TypedFeature("zobrist") {
            document_title("Map Based Hash Distribution");

            
            add_option<bool>("is_polynomial", "polynomial hashing.", "false");
            document_note("Polynomial Load Balancing",
            "Use polynomial hashing for underlying load balancing scheme");
        }

        virtual shared_ptr<ZobristHash> create_component(
            const plugins::Options &opts,
            const utils::Context &) const override {
            plugins::Options options_copy(opts);
            return plugins::make_shared_from_arg_tuples<ZobristHash>(
                options_copy.get<bool>("is_polynomial")
                );
        }
    };

    static plugins::FeaturePlugin<ZobristHashFeatures> _plugin_zobrist;



    class AdaptiveAbstractionHashFeatures
        : public plugins::TypedFeature<MapBasedHash, AdaptiveAbstractionHash> {
    public:
        AdaptiveAbstractionHashFeatures() : TypedFeature("aabstraction") {
            document_title("Map Based Hash Distribution");

            add_option<double>("ratio",
                    "abstraction ratio of actions to eliminate CO", "0.3");
            document_note(
                "Abstraction Ratio",
                "The maximum ratio of actions to eliminate CO."
                "If abst=0, then it is same as ZobristHash"
            );
            
            add_option<bool>("is_polynomial", "polynomial hashing.", "false");
            document_note("Polynomial Load Balancing",
            "Use polynomial hashing for underlying load balancing scheme");
        }

        virtual shared_ptr<AdaptiveAbstractionHash> create_component(
            const plugins::Options &opts,
            const utils::Context &) const override {
            plugins::Options options_copy(opts);
            return plugins::make_shared_from_arg_tuples<AdaptiveAbstractionHash>(
                options_copy.get<double>("ratio"),
                options_copy.get<bool>("is_polynomial")
                );
        }
    };
    static plugins::FeaturePlugin<AdaptiveAbstractionHashFeatures> _plugin_adaptive_abstraction;



    template<typename T>
    concept HasHashName = requires {
        { T::hash_name() } -> std::convertible_to<const char*>;
    };

    // Modified MapBasedHashPlugin
    template<HasHashName T>
        class MapBasedHashPlugin : public plugins::TypedFeature<MapBasedHash, T> {
        public:
            MapBasedHashPlugin() : plugins::TypedFeature<MapBasedHash, T>(T::hash_name()) {
                this->template document_title("Abstraction Map-based Hash Distribution");
                
                this->template add_option<int>("abstraction",
                        "actions to eliminate CO", "3");
                this->template document_note(
                    "Abstraction Ratio",
                    "The maximum ratio of actions to eliminate CO."
                    "If abst=0, then it is same as ZobristHash"
                );
                this->template add_option<bool>("is_polynomial", "polynomial hashing.", "false");
                this->template document_note("Polynomial Load Balancing",
                "Use polynomial hashing for underlying load balancing scheme");
            }

            virtual std::shared_ptr<T> create_component(
                const plugins::Options &opts,
                const utils::Context &) const override {
                plugins::Options options_copy(opts);
                return plugins::make_shared_from_arg_tuples<T>(
                    options_copy.get<int>("abstraction"),
                    options_copy.get<bool>("is_polynomial")
                );
            }
        };


        static plugins::FeaturePlugin<MapBasedHashPlugin<AbstractionHash>> _plugin_abstraction;
        static plugins::FeaturePlugin<MapBasedHashPlugin<FeatureBasedStructuredZobristHash>> _plugin_feature_structured_zobrist;
        static plugins::FeaturePlugin<MapBasedHashPlugin<ActionBasedStructuredZobristHash>> _plugin_action_structured_zobrist;

}
