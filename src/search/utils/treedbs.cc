#include "treedbs.h"
#include <algorithm>
#include <stack>

namespace utils {

TreeDBS::TreeDBS(size_t entryLength) : entryLength(entryLength), root(std::make_unique<Node>(false)) {
    if (entryLength == 0) throw std::invalid_argument("Entry length must be > 0");
}

void TreeDBS::insert(const std::vector<int>& vec) {
    if (vec.empty()) return;
    if (!validateLength(vec)) throw std::invalid_argument("Invalid vector length");

    if (vec.size() == entryLength) {
        size_t h = hashVector(vec);
        auto& child = root->children[h];

        if (!child) {
            child = std::make_unique<Node>(true);
            child->data = vec;
            itemCount++;
        } else if (child->isLeaf) {
            if (child->data != vec) {
                // Handle collision by expanding to subtree
                auto oldData = std::move(child->data);
                child->isLeaf = false;
                itemCount--; // Will be re-added as leaves

                // Reinsert both vectors
                insert(oldData);
                insert(vec);
            }
        }
        return;
    }

    // Iterative insertion for larger vectors
    std::stack<std::pair<Node*, std::vector<int>>> stack;
    stack.emplace(root.get(), vec);

    while (!stack.empty()) {
        auto [current, currentVec] = stack.top();
        stack.pop();

        auto [left, right] = splitVector(currentVec);
        std::array<std::vector<int>, 2> halves{left, right};

        for (auto& half : halves) {
            size_t h = hashVector(half);

            if (half.size() == entryLength) {
                auto& child = current->children[h];
                if (!child) {
                    child = std::make_unique<Node>(true);
                    child->data = half;
                    itemCount++;
                } else if (child->isLeaf) {
                    if (child->data != half) {
                        // Handle collision
                        auto oldData = std::move(child->data);
                        child->isLeaf = false;
                        itemCount--;

                        insert(oldData);
                        insert(half);
                    }
                }
            } else {
                auto& child = current->children[h];
                if (!child) {
                    child = std::make_unique<Node>(false);
                }
                stack.emplace(child.get(), half);
            }
        }
    }
}

bool TreeDBS::contains(const std::vector<int>& vec) const {
    if (vec.empty() || !validateLength(vec)) return false;

    std::stack<std::pair<const Node*, std::vector<int>>> stack;
    stack.emplace(root.get(), vec);

    while (!stack.empty()) {
        auto [current, currentVec] = stack.top();
        stack.pop();

        if (currentVec.size() == entryLength) {
            size_t h = hashVector(currentVec);
            auto it = current->children.find(h);
            if (it == current->children.end()) return false;

            if (it->second->isLeaf) {
                return it->second->data == currentVec;
            } else {
                // Collision subtree - need to search deeper
                stack.emplace(it->second.get(), currentVec);
            }
        } else {
            auto [left, right] = splitVector(currentVec);
            std::array<std::vector<int>, 2> halves{left, right};

            for (auto& half : halves) {
                size_t h = hashVector(half);
                auto it = current->children.find(h);
                if (it == current->children.end()) return false;
                stack.emplace(it->second.get(), half);
            }
        }
    }

    return false;
}

size_t TreeDBS::size() const { return itemCount; }

void TreeDBS::clear() {
    root = std::make_unique<Node>(false);
    itemCount = 0;
}

// Helper implementations remain the same as previous version
size_t TreeDBS::hashVector(const std::vector<int>& vec) const {
    std::size_t seed = vec.size();
    for (const auto& i : vec) seed ^= i + 0x9e3779b9 + (seed << 6) + (seed >> 2);
    return seed;
}

std::pair<std::vector<int>, std::vector<int>> TreeDBS::splitVector(const std::vector<int>& vec) const {
    size_t mid = vec.size() / 2;
    return {{vec.begin(), vec.begin() + mid}, {vec.begin() + mid, vec.end()}};
}

bool TreeDBS::validateLength(const std::vector<int>& vec) const {
    return vec.size() % entryLength == 0;
}
}