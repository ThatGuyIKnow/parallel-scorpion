#ifndef C_TREEDBS_H
#define C_TREEDBS_H

#include <vector>
#include <unordered_map>
#include <memory>
#include <stack>
#include <utility>

namespace utils {
#include <vector>
#include <unordered_map>
#include <memory>
#include <stack>

    class TreeDBS {
    public:
        explicit TreeDBS(size_t entryLength);

        void insert(const std::vector<int>& vec);
        bool contains(const std::vector<int>& vec) const;
        size_t size() const;
        void clear();

    private:
        struct Node {
            std::unordered_map<size_t, std::unique_ptr<Node>> children;
            std::vector<int> data; // Only used in leaf nodes
            bool isLeaf;

            explicit Node(bool leaf = false) : isLeaf(leaf) {}
        };

        const size_t entryLength;
        std::unique_ptr<Node> root;
        size_t itemCount = 0;

        size_t hashVector(const std::vector<int>& vec) const;
        std::pair<std::vector<int>, std::vector<int>> splitVector(const std::vector<int>& vec) const;
        bool validateLength(const std::vector<int>& vec) const;
    };
}
#endif // TREEDB_H
