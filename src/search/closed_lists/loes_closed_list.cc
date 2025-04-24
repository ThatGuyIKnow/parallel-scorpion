#include "loes_closed_list.h"

#include <bits/stdc++.h>
#include <cmath>

#include "../task_utils/task_properties.h"
#include "../tasks/root_task.h"
#include "../state_registry.h"
#include "../task_utils/successor_generator.h"
#include "../utils/rng.h"
#include "../per_state_information.h"
#include "../plugins/plugin.h"

namespace loes_closed_list{
template<class LoesType>
loes::MappedBitState LoesClosedList<LoesType>::state_to_bitstate(const State &state) const
{
    state.unpack();
    return loes::MappedBitState(state.get_unpacked_values(), bitmap);
}


template <class LoesType>
vector<size_t> LoesClosedList<LoesType>::min_entropy_bitorder(size_t num_samples, size_t max_sample_iterations) const
{
    unique_ptr<StateRegistry> sample_registry = make_unique<StateRegistry>(task_proxy);
    const successor_generator::SuccessorGenerator &successor_generator = successor_generator::g_successor_generators[task_proxy];
    OperatorsProxy operators = task_proxy.get_operators();
    utils::RandomNumberGenerator rng;

    State initial_state = sample_registry->get_initial_state(); 
    vector<StateID> id_samples = { initial_state.get_id() };
    int max_id = initial_state.get_id().get_value();

    for (size_t i = 0; (i < max_sample_iterations) && (id_samples.size() < num_samples); i++)
    {
        StateID id = *rng.choose(id_samples);
        
        State s = sample_registry->lookup_state(id);

        vector<OperatorID> applicable_ops;
        successor_generator.generate_applicable_ops(s, applicable_ops);
        if(applicable_ops.empty())
            continue;
        
        OperatorID op_id = *rng.choose(applicable_ops);
        OperatorProxy op = operators[op_id];
        State succ_state = sample_registry->get_successor_state(s, op);
        if(succ_state.get_id().get_value() > max_id){
            id_samples.push_back(succ_state.get_id());
            max_id = succ_state.get_id().get_value();
        }
    }
    vector<shared_ptr<loes::BitState>> sampled_bitstates;
    for (const StateID &id : id_samples)
    {
        State s = sample_registry->lookup_state(id);
        sampled_bitstates.push_back(make_shared<loes::MappedBitState>(state_to_bitstate(s)));
    }
    sample_registry.reset();
    id_samples.clear();

    using SubTree = vector<shared_ptr<loes::BitState>>;
    set<SubTree> initial_subtrees = { sampled_bitstates };
    sampled_bitstates.clear();
    set<SubTree> *curr_subtrees = &initial_subtrees;
    map<size_t, set<SubTree>> subtrees_if_p_next_bit = {};
    vector<size_t> bit_order = vector<size_t>(state_bit_length);
    set<size_t> unassigned_bit_positions = {};
    for (int i = 0; i < state_bit_length; i++)
    {
        unassigned_bit_positions.insert(i);
    }
    size_t curr_p = 0;
    size_t prev_best_p = 0;
    while (!unassigned_bit_positions.empty())
    {
        for (size_t p : unassigned_bit_positions)
        {
            subtrees_if_p_next_bit[p] = {};
            for (const SubTree &subtree : *curr_subtrees)
            {
                SubTree p_true = {};
                SubTree p_false = {};
                for (const auto &bitstate : subtree)
                {
                    if(bitstate->operator[](p)){
                        p_true.push_back(bitstate);
                    } else {
                        p_false.push_back(bitstate);
                    }
                }
                subtrees_if_p_next_bit[p].insert(p_true);
                subtrees_if_p_next_bit[p].insert(p_false);
            }
        }
        //Calculate entropy if p next bit
        double min_entropy = std::numeric_limits<double>::max();
        size_t best_p = state_bit_length;
        for (size_t p : unassigned_bit_positions)
        {
            double entropy = 0;
            set<SubTree> &subtrees = subtrees_if_p_next_bit[p];
            for (const auto &subtree : subtrees)
            {
                double p = (double) subtree.size() / num_samples;
                if(p > 0) entropy -= p * log(p);
            }
            
            if (entropy < min_entropy)
            {
                best_p = p;
                min_entropy = entropy;
            }
        }
        bit_order[best_p] = curr_p;
        curr_subtrees = &subtrees_if_p_next_bit[best_p];
        if(curr_p == 0)
            initial_subtrees.clear();
        else
            subtrees_if_p_next_bit.erase(prev_best_p);
        unassigned_bit_positions.erase(best_p);
        prev_best_p = best_p;
        ++curr_p;
    }
    
    return bit_order;
}
static int get_bit_size_for_range(int range) {
    int num_bits = 0;
    while ((1U << num_bits) < static_cast<unsigned int>(range))
        ++num_bits;
    return num_bits;
}

//TODO: Figure out how to get task_proxy through other means.
template<class LoesType>
LoesClosedList<LoesType>::LoesClosedList(const bool reuse_treelevels,
                                         const bool full_print,
                                         const int samples,
                                         const int max_sample_iterations) : task_proxy(*tasks::g_root_task), c_list(reuse_treelevels),
state_packer(task_properties::g_state_packers[task_proxy]), full_print(full_print)
{
    state_bit_length = 0;
    for (const auto &var : task_proxy.get_variables()) {
        assert(var.get_domain_size() > 1);
        int bit_size = get_bit_size_for_range(var.get_domain_size());
        state_bit_length += bit_size;
        var_bit_lengths.push_back((unsigned int) bit_size);    
    }
    bitmap = make_shared<loes::BitMap>(var_bit_lengths);

    int sample_size = samples;
    if (sample_size > 0)
    {
        vector<size_t> bit_order = min_entropy_bitorder((size_t) sample_size, max_sample_iterations);
        bitmap = make_shared<loes::BitMap>(var_bit_lengths, bit_order);
    }    
}

template<class LoesType>
void LoesClosedList<LoesType>::add_state(const State &state) {
    c_list.add_state(state_to_bitstate(state));
}

template<class LoesType>
bool LoesClosedList<LoesType>::contains_state(const State &state)
{
    return c_list.contains(state_to_bitstate(state));
}

template<class LoesType>
void LoesClosedList<LoesType>::print() const
{
    cout << "Closed list contains ";
    // c_list.print(full_print);
}

    void add_options_to_parser(plugins::Feature &feature) {
        feature.document_synopsis("Closed list using LOES data structure");
        feature.add_option<int>(
            "samples",
            "number of samples used to calculate optimal bitorder",
            "1000");
        feature.add_option<int>(
            "max_sample_iterations",
            "the max number of iterations tried to get the desired number of samples");
        feature.add_option<bool>(
            "reuse_treelevels",
            "whether LOES reuses treelevels when mergeing, taking more memory but less time.",
            "true");
        feature.add_option<bool>(
            "full_print",
            "will print() print the full LOES tree.",
            "false");
}

class LoesClosedListFeature
    : public plugins::TypedFeature<ClosedList, LoesClosedList<loes::Loes>> {
public:
    LoesClosedListFeature() : TypedFeature("loes") {
        document_title("LOES Closed List");

        add_options_to_parser(*this);
    }

    [[nodiscard]] shared_ptr<LoesClosedList<loes::Loes>>
    create_component(const plugins::Options &opts) const override {
        return plugins::make_shared_from_arg_tuples<LoesClosedList<loes::Loes>>(
            opts.get<int>("samples", 0),
            opts.get<int>("max_sample_iterations", 10 * opts.get<int>("samples", 0)),
            opts.get<bool>("reuse_treelevels", true),
            opts.get<bool>("full_print", false)
        );
    }
};

class CLoesClosedListFeature
    : public plugins::TypedFeature<ClosedList, LoesClosedList<loes::Cloes>> {
public:
    CLoesClosedListFeature() : TypedFeature("cloes") {
        document_title("CLOES Closed List");

        add_options_to_parser(*this);
    }

    shared_ptr<LoesClosedList<loes::Cloes>>
    create_component(const plugins::Options &opts) const override {
        
        return plugins::make_shared_from_arg_tuples<LoesClosedList<loes::Cloes>>(
            opts.get<bool>("reuse_treelevels", true),
            opts.get<bool>("full_print", false),
            opts.get<int>("samples", 0),
            opts.get<int>("max_sample_iterations", 10 * opts.get<int>("samples", 0))
        );
    }
};


static plugins::FeaturePlugin<LoesClosedListFeature> _loes_plugin;
[[maybe_unused]] static plugins::FeaturePlugin<CLoesClosedListFeature> _cloes_plugin;
}
