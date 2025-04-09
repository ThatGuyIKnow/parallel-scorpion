#include "treedbs.h"

#include <queue>
#include <span>

#include "hash.h"
#include <stack>
#include <stdexcept>

#include "logging.h"

namespace utils {


    TreeDBS::TreeDBS(size_t size) : _size(size) {
        assert(size > 0 && "TreeDBS size must be greater than 0");
        root = constructTreeHelper(size);
    }

    std::unique_ptr<TreeDBS::Node> TreeDBS::constructTreeHelper(size_t input_size, int level) {
        auto node = std::make_unique<Node>();
        node ->level = level;

        if (input_size <= 2) {
            node->isLeaf = true;
            return node;
        }

        auto mid = splitRange(0, input_size);
        assert(mid > 0 && mid < input_size && "Invalid split range");

        auto left= constructTreeHelper(mid, level+1);
        auto right = constructTreeHelper(input_size - mid, level+1);
        node->isLeaf = false;
        node->left = std::move(left);
        node->right = std::move(right);

        return node;
    }

    void TreeDBS::insert(const std::vector<int>& vec) {
        assert(vec.size() == _size && "Vector size must match TreeDBS size");
        putRecursively(vec.begin(), vec.end(), root);
        //findOrPutIterative(vec, true);

    }

    bool TreeDBS::contains(const std::vector<int>& vec) {
        assert(vec.size() == _size && "Vector size must match TreeDBS size");

        return findRecursively(vec.begin(), vec.end(), root) >= 0;
    }

    size_t TreeDBS::size() const {
        return root->entries.size();
    }


    int TreeDBS::findRecursively(std::vector<int>::const_iterator begin, std::vector<int>::const_iterator end, const std::unique_ptr<Node>& node) {
        assert(!node->isLeaf || std::distance(begin, end) <= 2 && "Leaf node must have 2 or fewer elements");

        if (node->isLeaf) {
            return node->values.find(std::vector(begin, end));
        }

        const auto mid = begin + splitRange(0, std::distance(begin, end));
        assert(mid > begin && mid < end && "Invalid midpoint calculation");


        auto left_index = findRecursively(begin, mid, node->left);
        if (left_index == -1)
            return -1;

        auto right_index = findRecursively(mid, end, node->right);
        if (right_index == -1)
            return -1;

        auto entry = std::vector<int>{left_index, right_index};
        return node->entries.find(entry);
    }

    int TreeDBS::putRecursively(const std::vector<int>::const_iterator begin, const std::vector<int>::const_iterator end, const std::unique_ptr<Node>& node) {
        assert(!node->isLeaf || std::distance(begin, end) <= 2 && "Leaf node must have 2 or fewer elements");
        if (node->isLeaf) {
            return node->values.find_or_insert(std::vector<int>{begin, end});
        }

        auto mid = begin + splitRange(0, std::distance(begin, end));
        assert(mid > begin && mid < end && "Invalid midpoint calculation");

        auto left_index = putRecursively(begin, mid, node->left);
        auto right_index = putRecursively(mid, end, node->right);

        auto entry = std::vector<int>{left_index, right_index};
        return node->entries.find_or_insert(entry);
    }


    void TreeDBS::print_info() const {
        root->entries.print_info();
        root->values.print_info();
    }

} // namespace utils
