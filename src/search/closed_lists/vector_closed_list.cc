#include "vector_closed_list.h"
#include "../plugins/plugin.h"
#include "../task_utils/task_properties.h"
#include "../task_proxy.h"
#include "../tasks/root_task.h"
#include "../state_id.h"


namespace vector_closed_list {
    VectorClosedList::VectorClosedList()
    : task_proxy(*tasks::g_root_task),
    state_packer(task_properties::g_state_packers[task_proxy]),
      registered_states(
          0,
          StateIDSemanticHash(get_bins_per_state()),
          StateIDSemanticEqual(get_bins_per_state())){

    }
    void VectorClosedList::add_state(const State &state) {
        int num_bins = get_bins_per_state();
        std::vector<PackedStateBin> bins(num_bins, 0);

        for (int i = 0; i < state.size(); ++i)
            state_packer.set(bins.data(), i, state[i].get_value());

        registered_states.insert(bins);
    }


    bool VectorClosedList::contains_state(const State &state) {
        int num_bins = get_bins_per_state();
        std::vector<PackedStateBin> bins(num_bins, 0);

        for (int i = 0; i < state.size(); ++i)
            state_packer.set(bins.data(), i, state[i].get_value());

        return registered_states.find(bins) != registered_states.end();
    }

    int VectorClosedList::get_bins_per_state() const {
        return state_packer.get_num_bins();
    }

    void VectorClosedList::print() const {
        utils::g_log << "Registry size: " << registered_states.size() << std::endl;
    }

    class VectorClosedListFeature
        : public plugins::TypedFeature<ClosedList, VectorClosedList> {
    public:
        VectorClosedListFeature() : TypedFeature("vector") {
            document_title("Vector Closed List");
        }

        virtual std::shared_ptr<VectorClosedList>
        create_component(const plugins::Options &opts) const override {

            return plugins::make_shared_from_arg_tuples<VectorClosedList>();
        }
    };

    static plugins::FeaturePlugin<VectorClosedListFeature> _plugin;


}