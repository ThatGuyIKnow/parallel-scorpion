#include "treedbs_closed_list.h"
#include "../plugins/plugin.h"
#include "../task_utils/task_properties.h"
#include "../task_proxy.h"
#include "../tasks/root_task.h"



namespace treedbs_closed_list {

    TreeDBSClosedList::TreeDBSClosedList() : task_proxy(*tasks::g_root_task){
        dbs = std::make_shared<utils::TreeDBS>(task_proxy.get_variables().size());
    }


    void TreeDBSClosedList::add_state(const State &state) {
        state.unpack();
        const std::vector<int> entry = state.get_unpacked_values();
        return dbs->insert(entry);
    }

    bool TreeDBSClosedList::contains_state(const State &state) const {
        state.unpack();
        const std::vector<int>& entry = state.get_unpacked_values();
        return dbs->contains(entry);
    }

    void TreeDBSClosedList::print() const {
        utils::g_log << "Registry size: " << dbs->size() << std::endl;
    }



    class TreeDBSClosedListFeature
        : public plugins::TypedFeature<ClosedList, TreeDBSClosedList> {
    public:
        TreeDBSClosedListFeature() : TypedFeature("treedbs") {
            document_title("Tree Database Closed List");

        }

        virtual std::shared_ptr<TreeDBSClosedList>
        create_component(const plugins::Options &opts) const override {

            return plugins::make_shared_from_arg_tuples<TreeDBSClosedList>();
        }
    };

    static plugins::FeaturePlugin<TreeDBSClosedListFeature> _plugin;

} // treedbs_closed_list