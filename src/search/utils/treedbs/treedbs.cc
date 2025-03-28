#include "treedbs.h"
#include <algorithm>
#include <stack>

TreeDB::TreeDB(size_t entryLength) : entryLength(entryLength), root(std::make_unique<Node>(false)) {
    if (entryLength == 0) {
        throw std::invalid_argument("Entry length must be greater than 0");
    }
}

TreeDB::~TreeDB() = default;

size_t TreeDB::hashVector(const std::vector<int>& vec) const {
    std::size_t seed = vec.size();
    for (const auto& i : vec) {
        seed ^= i + 0x9e3779b9 + (seed << 6) + (seed >> 2);
    }
    return seed;
}

std::vector<std::vector<int>> TreeDB::splitVector(const std::vector<int>& vec) const {
    std::vector<std::vector<int>> chunks;
    for (size_t i = 0; i < vec.size(); i += entryLength) {
        auto last = std::min(vec.size(), i + entryLength);
        chunks.emplace_back(vec.begin() + i, vec.begin() + last);
    }
    return chunks;
}

bool TreeDB::validateLength(const std::vector<int>& vec) const {
    return vec.size() % entryLength == 0;
}

void TreeDB::insert(const std::vector<int>& vec) {
    if (vec.empty()) return;
    if (!validateLength(vec)) {
        throw std::invalid_argument("Vector length must be a multiple of entry length");
    }

    if (vec.size() == entryLength) {
        size_t h = hashVector(vec);
        auto& child = root->children[h];
        if (!child) {
            child = std::make_unique<Node>(true);
            child->data = vec;
        }
        return;
    }

    // Iterative insertion using a stack
    std::stack<std::pair<Node*, std::vector<int>>> stack;
    stack.emplace(root.get(), vec);

    while (!stack.empty()) {
        auto [currentNode, currentVec] = stack.top();
        stack.pop();

        auto chunks = splitVector(currentVec);

        for (const auto& chunk : chunks) {
            size_t h = hashVector(chunk);

            if (currentNode->children.find(h) == currentNode->children.end()) {
                bool isLeaf = (chunk.size() == entryLength);
                currentNode->children[h] = std::make_unique<Node>(isLeaf);
                if (isLeaf) {
                    currentNode->children[h]->data = chunk;
                }
            }

            if (chunk.size() > entryLength) {
                stack.emplace(currentNode->children[h].get(), chunk);
            }
        }
    }
}

bool TreeDB::contains(const std::vector<int>& vec) const {
    if (vec.empty()) return false;
    if (!validateLength(vec)) return false;

    if (vec.size() == entryLength) {
        size_t h = hashVector(vec);
        auto it = root->children.find(h);
        return it != root->children.end() && it->second->isLeaf && it->second->data == vec;
    }

    // Iterative search using a stack
    std::stack<std::pair<const Node*, std::vector<int>>> stack;
    stack.emplace(root.get(), vec);

    while (!stack.empty()) {
        auto [currentNode, currentVec] = stack.top();
        stack.pop();

        auto chunks = splitVector(currentVec);
        bool allChunksValid = true;

        for (const auto& chunk : chunks) {
            size_t h = hashVector(chunk);
            auto it = currentNode->children.find(h);

            if (it == currentNode->children.end()) {
                allChunksValid = false;
                break;
            }

            if (chunk.size() == entryLength) {
                if (!it->second->isLeaf || it->second->data != chunk) {
                    allChunksValid = false;
                    break;
                }
            } else {
                stack.emplace(it->second.get(), chunk);
            }
        }

        if (!allChunksValid) {
            return false;
        }
    }

    return true;
}

size_t TreeDB::size() const {
    // Iterative count of all leaf nodes using a stack
    size_t count = 0;
    std::stack<const Node*> stack;
    stack.push(root.get());

    while (!stack.empty()) {
        const Node* current = stack.top();
        stack.pop();

        if (current->isLeaf) {
            count++;
        } else {
            for (const auto& child : current->children) {
                stack.push(child.second.get());
            }
        }
    }

    return count;
}

void TreeDB::clear() {
    root = std::make_unique<Node>(false);
}
