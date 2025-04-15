#include "unpacked_closed_list.h"
#include "../plugins/plugin.h"
#include "../task_utils/task_properties.h"
#include "../task_proxy.h"
#include "../tasks/root_task.h"
#include "../state_id.h"


namespace unpacked_closed_list {
    UnpackedClosedList::UnpackedClosedList()
    : task_proxy(*tasks::g_root_task),
      num_variables(task_proxy.get_variables().size()),
      registered_states(
          0,
          StateIDSemanticHash(num_variables),
          StateIDSemanticEqual(num_variables)){

    }
    void UnpackedClosedList::add_state(const State &state) {
        std::vector<int> state_variables;
        state_variables.reserve(num_variables);
        for (auto fact : state) {
            state_variables.push_back(fact.get_value());
        }
        registered_states.insert(state_variables);

    }


    bool UnpackedClosedList::contains_state(const State &state) {
        std::vector<int> state_variables;
        state_variables.reserve(num_variables);
        for (auto fact : state) {
            state_variables.push_back(fact.get_value());
        }
        return registered_states.contains(state_variables);
    }


    void UnpackedClosedList::print() const {
        utils::g_log << "Registry size: " << registered_states.size() << std::endl;
    }

    class UnpackedClosedListFeature
        : public plugins::TypedFeature<ClosedList, UnpackedClosedList> {
    public:
        UnpackedClosedListFeature() : TypedFeature("unpacked") {
            document_title("Unpacked Vector Closed List");
        }

        virtual std::shared_ptr<UnpackedClosedList>
        create_component(const plugins::Options &opts) const override {

            return plugins::make_shared_from_arg_tuples<UnpackedClosedList>();
        }
    };

    static plugins::FeaturePlugin<UnpackedClosedListFeature> _plugin;


}