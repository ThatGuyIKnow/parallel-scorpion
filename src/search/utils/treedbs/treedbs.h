#ifndef C_TREEDBS_H
#define C_TREEDBS_H

#include <vector>
#include <unordered_map>
#include <memory>
#include <stack>
#include <utility>

class TreeDB {
public:
    explicit TreeDB(size_t entryLength);
    ~TreeDB();

    void insert(const std::vector<int>& vec);
    bool contains(const std::vector<int>& vec) const;
    size_t size() const;
    void clear();

private:
    struct Node {
        std::unordered_map<size_t, std::unique_ptr<Node>> children;
        std::vector<int> data;
        bool isLeaf;

        explicit Node(bool leaf = false) : isLeaf(leaf) {}
    };

    const size_t entryLength;
    std::unique_ptr<Node> root;

    size_t hashVector(const std::vector<int>& vec) const;
    std::vector<std::vector<int>> splitVector(const std::vector<int>& vec) const;
    bool validateLength(const std::vector<int>& vec) const;
};
#endif