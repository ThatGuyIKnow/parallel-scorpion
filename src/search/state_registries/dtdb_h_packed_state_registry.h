#ifndef DTDB_H_PACKED_STATE_REGISTRY_H
#define DTDB_H_PACKED_STATE_REGISTRY_H

#include "../abstract_task.h"
#include "../axioms.h"
#include "../state_id.h"
#include "../state_registry.h"

#include "../algorithms/int_packer.h"
#include "../algorithms/segmented_vector.h"
#include "../algorithms/subscriber.h"
#include "../utils/hash.h"

#include <parallel_hashmap/phmap.h>

#include <vector>

#include "../task_utils/task_properties.h"
#include "../task_proxy.h"

#include "../ext/valla/dtdb_h.hpp"

class DtdbHPackedStateRegistry : public StateRegistry {
    using PackedStateBin = int_packer::IntPacker::Bin;

    int_packer::IntPacker &state_packer;
    AxiomEvaluator &axiom_evaluator;
    const int num_variables;

    // DTDB_H storage/dedup:
    valla::DtdbHTable table_;
    std::vector<valla::Index> state_roots_;                  // StateID -> root index
    phmap::flat_hash_map<valla::Index, int> root_to_state_;  // root index -> StateID

    // Packed data pool to expose buffers (like PackedStateRegistry)
    segmented_vector::SegmentedArrayVector<PackedStateBin> state_data_pool;

    std::unique_ptr<State> cached_initial_state;

    int get_bins_per_state() const { return state_packer.get_num_bins(); }

    StateID insert_or_get_id(valla::Index root) {
        auto [it, inserted] = root_to_state_.emplace(root, static_cast<int>(state_roots_.size()));
        if (inserted) {
            state_roots_.push_back(root);
            return StateID(it->second);
        } else {
            // Drop last pushed packed buffer if we inserted one preemptively (we push only on new)
            return StateID(it->second);
        }
    }

    void rebuild_packed_buffer_from_root(PackedStateBin *buffer, valla::Index root) const {
        std::vector<valla::Index> seq;
        seq.reserve(num_variables);
        valla::read_sequence(root, table_, std::back_inserter(seq));
        for (int i = 0; i < num_variables; ++i) {
            state_packer.set(buffer, i, static_cast<int>(seq[i]));
        }
    }

public:
    explicit DtdbHPackedStateRegistry(const TaskProxy &task_proxy)
        : StateRegistry(task_proxy),
          state_packer(task_properties::g_state_packers[task_proxy]),
          axiom_evaluator(g_axiom_evaluators[task_proxy]),
          num_variables(task_proxy.get_variables().size()),
          table_(), state_roots_(), root_to_state_(),
          state_data_pool(get_bins_per_state()),
          cached_initial_state(nullptr) {
        State::get_variable_value = [this](const StateID &id) {
            std::vector<int> unpacked(num_variables, 0);
            std::vector<valla::Index> seq;
            seq.reserve(num_variables);
            valla::read_sequence(state_roots_[id.get_value()], table_, std::back_inserter(seq));
            for (int i = 0; i < num_variables; ++i) unpacked[i] = static_cast<int>(seq[i]);
            return unpacked;
        };
    }

    const TaskProxy &get_task_proxy() const override { return task_proxy; }
    int get_num_variables() const override { return num_variables; }
    const int_packer::IntPacker &get_state_packer() const override { return state_packer; }

    State lookup_state(StateID id) const override {
        return task_proxy.create_state(*this, id);
    }

    State lookup_state(StateID id, std::vector<int> &&state_values) const override {
        return task_proxy.create_state(*this, id, std::move(state_values));
    }

    const State &get_initial_state() override {
        if (!cached_initial_state) {
            std::vector<valla::Index> seq;
            seq.reserve(num_variables);
            State initial_state = task_proxy.get_initial_state();
            for (size_t i = 0; i < initial_state.size(); ++i)
                seq.push_back(static_cast<valla::Index>(initial_state[i].get_value()));
            const auto root = valla::insert_sequence(seq, table_);

            // push packed buffer
            std::unique_ptr<PackedStateBin[]> buffer(new PackedStateBin[get_bins_per_state()]);
            std::fill_n(buffer.get(), get_bins_per_state(), 0);
            rebuild_packed_buffer_from_root(buffer.get(), root);
            state_data_pool.push_back(buffer.get());

            StateID id = insert_or_get_id(root);
            cached_initial_state = std::make_unique<State>(lookup_state(id));
        }
        return *cached_initial_state;
    }

    State get_successor_state(const State &predecessor, const OperatorProxy &op) override {
        // unpack predecessor, apply effects, axioms
        predecessor.unpack();
        std::vector<int> new_values = predecessor.get_unpacked_values();
        for (EffectProxy effect : op.get_effects()) {
            if (does_fire(effect, predecessor)) {
                FactPair effect_pair = effect.get_fact().get_pair();
                new_values[effect_pair.var] = effect_pair.value;
            }
        }
        if (task_properties::has_axioms(task_proxy))
            axiom_evaluator.evaluate(new_values);

        // DTDB insert
        std::vector<valla::Index> seq;
        seq.reserve(num_variables);
        for (int v : new_values) seq.push_back(static_cast<valla::Index>(v));
        const auto root = valla::insert_sequence(seq, table_);

        // build packed buffer for new state
    state_data_pool.push_back(state_data_pool[predecessor.get_id().get_value()]);
        PackedStateBin *buffer = state_data_pool[state_data_pool.size() - 1];
        rebuild_packed_buffer_from_root(buffer, root);

        StateID id = insert_or_get_id(root);
        return lookup_state(id);
    }

    size_t size() const override { return state_roots_.size(); }

    void print_statistics(utils::LogProxy &log) const override {
        log << "Number of registered states: " << size() << std::endl;
    }

    // Iteration over 0..size()-1
    class iterator_impl : public IStateRegistry::const_iterator {
        const DtdbHPackedStateRegistry *registry_;
        size_t idx_;
    public:
        iterator_impl(const DtdbHPackedStateRegistry *reg, size_t i) : registry_(reg), idx_(i) {}
        StateID operator*() const override { return StateID(idx_); }
        const_iterator &operator++() override { ++idx_; return *this; }
        bool operator==(const const_iterator &other) const override {
            auto p = dynamic_cast<const iterator_impl*>(&other);
            return p && registry_ == p->registry_ && idx_ == p->idx_;
        }
        std::unique_ptr<const_iterator> clone() const override {
            return std::make_unique<iterator_impl>(registry_, idx_);
        }
    };

    std::unique_ptr<IStateRegistry::const_iterator> begin() const override {
        return std::make_unique<iterator_impl>(this, 0);
    }
    std::unique_ptr<IStateRegistry::const_iterator> end() const override {
        return std::make_unique<iterator_impl>(this, size());
    }
};

#endif // DTDB_H_PACKED_STATE_REGISTRY_H
