#include "treedbs.h"

namespace utils {

    TreeDBSAdapter::TreeDBSAdapter(const int len) : dbs(TreeDBScreate(len)) {}

    void TreeDBSAdapter::insert(std::vector<int> entry) {
        TreeDBSlookup(dbs, entry.data());
    }

    bool TreeDBSAdapter::lookup(std::vector<int> entry) const {
        return TreeDBSContains(dbs, entry.data()) >= 0;
    }
    int TreeDBSAdapter::get(const int index, const int pos) const {
        return TreeDBSGet(dbs, index, pos);
    }
    int TreeDBSAdapter::count() const {
        return TreeCount(dbs);
    }
    void TreeDBSAdapter::info() const {
        TreeInfo(dbs);
    }
    void TreeDBSAdapter::clear() const {
        TreeDBSclear(dbs);
    }

    void TreeDBSAdapter::stats() const {
        TreeDBSstats(dbs);
    }




} // utils