#ifndef TREE_UNPACKED_STATE_REGISTRY_H
#define TREE_UNPACKED_STATE_REGISTRY_H

#include "../abstract_task.h"
#include "../axioms.h"
#include "../state_id.h"
#include "../state_registry.h"

#include "../algorithms/int_packer.h"
#include "../algorithms/segmented_vector.h"
#include "../algorithms/subscriber.h"
#include "../utils/hash.h"

#include "valla_adapter.h"

#include <parallel_hashmap/phmap.h>
#include <set>

namespace utils {
class LogProxy;
}

using IStateRegistry = StateRegistry;
class TreeUnpackedStateRegistry :
    public IStateRegistry {

    vs::IndexedHashSet<std::vector<int>> tree_table;

    const int_packer::IntPacker &state_packer;
    AxiomEvaluator &axiom_evaluator;
    const int num_variables;

    std::unique_ptr<State> cached_initial_state;

    size_t _registered_states = 0;
    StateID insert_id_or_pop_state();
    int get_bins_per_state() const;
public:
    explicit TreeUnpackedStateRegistry(const TaskProxy &task_proxy);

    const TaskProxy &get_task_proxy() const override {
        return task_proxy;
    }

    int get_num_variables() const override {
        return num_variables;
    }

    const int_packer::IntPacker &get_state_packer() const override {
        return state_packer;
    }

    /*
      Returns the state that was registered at the given ID. The ID must refer
      to a state in this registry. Do not mix IDs from from different registries.
    */
    State lookup_state(StateID id) const override;

    /*
      Like lookup_state above, but creates a state with unpacked data,
      moved in via state_values. It is the caller's responsibility that
      the unpacked data matches the state's data.
    */
    State lookup_state(StateID id, std::vector<int> &&state_values) const override;

    /*
      Returns a reference to the initial state and registers it if this was not
      done before. The result is cached internally so subsequent calls are cheap.
    */
    const State &get_initial_state() override;

    /*
      Returns the state that results from applying op to predecessor and
      registers it if this was not done before. This is an expensive operation
      as it includes duplicate checking.
    */
    State get_successor_state(const State &predecessor, const OperatorProxy &op) override;

    /*
      Returns the number of states registered so far.
    */
    size_t size() const override {
        return _registered_states;
    }

    int get_state_size_in_bytes() const;

    void print_statistics(utils::LogProxy &log) const override;

    class const_iterator {
        using iterator_category = std::forward_iterator_tag;
        using value_type = StateID;
        using difference_type = ptrdiff_t;
        using pointer = StateID *;
        using reference = StateID &;

        /*
          We intentionally omit parts of the forward iterator concept
          (e.g. default construction, copy assignment, post-increment)
          to reduce boilerplate. Supported compilers may complain about
          this, in which case we will add the missing methods.
        */

        friend class TreeUnpackedStateRegistry;
        const TreeUnpackedStateRegistry &registry;
        StateID pos;

        const_iterator(const TreeUnpackedStateRegistry &registry, size_t start)
            : registry(registry), pos(start) {
            utils::unused_variable(this->registry);
        }
public:
        const_iterator &operator++() {
            ++pos.value;
            return *this;
        }

        bool operator==(const const_iterator &rhs) const {
            assert(&registry == &rhs.registry);
            return pos == rhs.pos;
        }

        bool operator!=(const const_iterator &rhs) const {
            return !(*this == rhs);
        }

        StateID operator*() {
            return pos;
        }

        StateID *operator->() {
            return &pos;
        }
    };
    class iterator_impl : public IStateRegistry::const_iterator {
        const TreeUnpackedStateRegistry *registry_;
        size_t idx_;
    public:
        iterator_impl(const TreeUnpackedStateRegistry *reg, size_t i) : registry_(reg), idx_(i) {}
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

#endif
