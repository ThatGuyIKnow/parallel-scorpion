#ifndef TREEDBS_H
#define TREEDBS_H

#include <memory>
#include <vector>
#include "treedbs/treedbs.h"

namespace utils {

    class TreeDBSAdapter {
            treedbs_s* dbs;  // Pointer to the C struct

    public:
        explicit TreeDBSAdapter(int len);
        void insert(std::vector<int> entry);
        bool lookup(std::vector<int> entry) const;
        int get(int index, int pos) const;
        int count() const;
        void info() const;
        void clear() const;
        void stats() const;
    };

}

#endif // TREEDBS_H