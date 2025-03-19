#ifndef STATE_REGISTRY_H
#define STATE_REGISTRY_H

#include "../abstract_task.h"
#include "../axioms.h"
#include "../state_id.h"

#include "../algorithms/int_hash_set.h"
#include "../algorithms/int_packer.h"
#include "../algorithms/segmented_vector.h"
#include "../algorithms/subscriber.h"
#include "../utils/hash.h"

#include <parallel_hashmap/phmap.h>

#include <set>
#include <vector>

using PackedStateBin = int_packer::IntPacker::Bin;

namespace int_packer {
  class IntPacker;
  };
  
class StateRegistry : public subscriber::SubscriberService<StateRegistry> {
protected:
    TaskProxy task_proxy;
    const int_packer::IntPacker &state_packer;
    AxiomEvaluator &axiom_evaluator;
    const int num_variables;

    std::unique_ptr<State> cached_initial_state;

    int get_bins_per_state() const;
    static int get_bins_per_state(int_packer::IntPacker state_packer);

public:
    explicit StateRegistry(const TaskProxy &task_proxy);
    virtual ~StateRegistry() = default;

    const TaskProxy &get_task_proxy() const;
    int get_num_variables() const;
    const int_packer::IntPacker &get_state_packer() const;

    virtual const State &get_initial_state() = 0; // Pure virtual
    int get_state_size_in_bytes() const;
    virtual void print_statistics(utils::LogProxy &log) const;

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

        friend class StateRegistry;
        const StateRegistry &registry;
        StateID pos;

        const_iterator(const StateRegistry &registry, size_t start)
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
    
    
    const_iterator begin() const {
      return const_iterator(*this, 0);
    }

    const_iterator end() const {
        return const_iterator(*this, size());
    }

    
    virtual StateID insert_buffered_state(const PackedStateBin *buffer) = 0;
    virtual State lookup_state(StateID id) const = 0;
    virtual State lookup_state(StateID id, std::vector<int> &&state_values) const = 0;
    virtual State get_successor_state(const State &predecessor, const OperatorProxy &op) = 0;
    virtual size_t size() const = 0;
    virtual void print_statistics(utils::LogProxy &log);

};


#endif
