#include "packed_treedbs_closed_list.h"
#include "../plugins/plugin.h"
#include "../task_utils/task_properties.h"
#include "../task_proxy.h"
#include "../tasks/root_task.h"
#include "../state_id.h"



namespace packed_treedbs_closed_list {

    PackedTreeDBSClosedList::PackedTreeDBSClosedList() : task_proxy(*tasks::g_root_task),
    state_packer(task_properties::g_state_packers[task_proxy]){
        dbs = std::make_unique<utils::TreeDBS>(state_packer.get_num_bins());
    }

    void PackedTreeDBSClosedList::add_state(const State &state) {
        const auto num_bins = state_packer.get_num_bins();
        std::unique_ptr<PackedStateBin[]> buffer(new PackedStateBin[num_bins]);
        // Avoid garbage values in half-full bins.
        std::fill_n(buffer.get(), num_bins, 0);

        for (size_t i = 0; i < state.size(); ++i) {
            state_packer.set(buffer.get(), i, state[i].get_value());
        }
        const std::vector<int> entry = {buffer.get(), buffer.get() + num_bins};
        return dbs->insert(entry);
    }

    bool PackedTreeDBSClosedList::contains_state(const State &state) {
        const auto num_bins = state_packer.get_num_bins();

        std::unique_ptr<PackedStateBin[]> buffer(new PackedStateBin[num_bins]);
        // Avoid garbage values in half-full bins.
        std::fill_n(buffer.get(), num_bins, 0);

        for (size_t i = 0; i < state.size(); ++i) {
            state_packer.set(buffer.get(), i, state[i].get_value());
        }
        const std::vector<int> entry = {buffer.get(), buffer.get() + num_bins};
        return dbs->contains(entry);
    }

    void PackedTreeDBSClosedList::print() const {
        utils::g_log << "Registry size: " << dbs->size() << std::endl;
        dbs->print_info();
    }

    class PackedTreeDBSClosedListFeature
        : public plugins::TypedFeature<ClosedList, PackedTreeDBSClosedList> {
    public:
        PackedTreeDBSClosedListFeature() : TypedFeature("ptreedbs") {
            document_title("Tree Database Closed List");
        }

        virtual std::shared_ptr<PackedTreeDBSClosedList>
        create_component(const plugins::Options &opts) const override {
            (void)opts;
            return plugins::make_shared_from_arg_tuples<PackedTreeDBSClosedList>();
        }
    };

    static plugins::FeaturePlugin<PackedTreeDBSClosedListFeature> _plugin;

} // packed_treedbs_closed_list