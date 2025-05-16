#pragma once
#include <vector>
#include <unordered_map>
#include <cmath>
#include <memory>
#include <limits>
#include <algorithm>
#include <numeric>

namespace mi_detail {
struct BlockHash {
    size_t operator()(const std::vector<int>& b) const {
        size_t hash = b.size();
        for (auto v : b)
            hash ^= std::hash<int>{}(v) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
        return hash;
    }
};
struct BlockEq {
    bool operator()(const std::vector<int>& a, const std::vector<int>& b) const { return a == b; }
};
// Canonical pair for block pairs (unordered)
struct BlockPair {
    std::vector<int> first, second;
    BlockPair(const std::vector<int>& a, const std::vector<int>& b) {
        if (a < b) { first = a; second = b; }
        else       { first = b; second = a; }
    }
    bool operator==(const BlockPair& other) const {
        return first == other.first && second == other.second;
    }
};
struct BlockPairHash {
    size_t operator()(const BlockPair& p) const {
        // Cantor pairing-like: mix two block hashes for symmetry
        size_t h1 = BlockHash{}(p.first), h2 = BlockHash{}(p.second);
        return h1 ^ (h2 + 0x9e3779b9 + (h1 << 6) + (h1 >> 2));
    }
};

template<typename T>
struct VecHash {
    size_t operator()(const std::vector<T>& v) const {
        size_t h = 0xcbf29ce484222325ULL;
        for (const auto& x : v)
            h = (h ^ std::hash<T>{}(x)) * 0x100000001b3ULL;
        return h;
    }
};

template<typename T>
using CountMap = std::unordered_map<std::vector<T>, int, VecHash<T>>;

// Global caches for entropy, block counts, and MI!
template <typename T>
struct Cache {
    static std::unordered_map<std::vector<int>, double, BlockHash, BlockEq> entropy_cache;
};

template <typename T>
std::unordered_map<std::vector<int>, double, BlockHash, BlockEq> Cache<T>::entropy_cache;


template<typename T>
std::shared_ptr<CountMap<T>> cached_counts(const std::vector<std::vector<T>>& states, const std::vector<int>& block) {
    std::vector<int> b = block; std::sort(b.begin(), b.end());
    auto counts = std::make_shared<CountMap<T>>();
    std::vector<T> key(b.size());
    for (const auto& row : states) {
        for (size_t j = 0; j < b.size(); ++j) key[j] = row[b[j]];
        ++(*counts)[key];
    }
    return counts;
}

template<typename T>
double cached_entropy(const std::vector<std::vector<T>>& states, const std::vector<int>& block) {
    std::vector<int> b = block; std::sort(b.begin(), b.end());
    auto& ec = Cache<T>::entropy_cache;
    if (auto it = ec.find(b); it != ec.end()) return it->second;
    auto counts = cached_counts(states, b);
    const double total = double(states.size());
    double ent = 0.0;
    for (const auto& kv : *counts) {
        double p = kv.second / total;
        ent -= p * std::log2(p);
    }
    ec[b] = ent;
    return ent;
}

inline std::vector<int> concat(const std::vector<int>& a, const std::vector<int>& b) {
    std::vector<int> out = a;
    out.insert(out.end(), b.begin(), b.end());
    return out;
}
} // namespace mi_detail

inline double entropySimd(const std::vector<std::vector<float>>& cols) {
    std::vector<std::vector<float>> fake_states = cols;
    std::vector<int> block(cols[0].size());
    std::iota(block.begin(), block.end(), 0);
    return mi_detail::cached_entropy(fake_states, block);
}

template<typename T>
double mutualInformationSimd(const std::vector<std::vector<T>>& states,
                             const std::vector<int>& block1, const std::vector<int>& block2) {
    using namespace mi_detail;
    BlockPair key(block1, block2);
    double H_X = cached_entropy(states, block1);
    double H_Y = cached_entropy(states, block2);
    double H_XY = cached_entropy(states, concat(block1, block2));
    double result = H_X + H_Y - H_XY;
    return result;
}

template<typename T>
std::vector<int> greedyMiMaxVariableOrderSimd(const std::vector<std::vector<T>>& states) {
    using namespace mi_detail;
    if (states.empty()) return {};
    std::vector<int> variables;
    for (size_t i = 0; i < states[0].size(); ++i) variables.push_back(static_cast<int>(i));

    std::vector<std::vector<int>> blocks;
    for (int v : variables) blocks.push_back({v});

    while (blocks.size() > 1) {
        double bestMi = -1;
        std::pair<int, int> bestPair{-1, -1};
        for (size_t i = 0; i < blocks.size(); ++i) {
            for (size_t j = i + 1; j < blocks.size(); ++j) {
                double mi = mutualInformationSimd(states, blocks[i], blocks[j]);
                if (mi > bestMi) {
                    bestMi = mi;
                    bestPair = {static_cast<int>(i), static_cast<int>(j)};
                }
            }
        }
        int i = bestPair.first, j = bestPair.second;
        std::vector<int> mergedBlock = concat(blocks[i], blocks[j]);
        std::vector<std::vector<int>> newBlocks;
        for (size_t k = 0; k < blocks.size(); ++k)
            if (static_cast<int>(k) != i && static_cast<int>(k) != j)
                newBlocks.push_back(blocks[k]);
        newBlocks.push_back(std::move(mergedBlock));
        blocks = std::move(newBlocks);
    
     }
    
    // --- Clear all caches for reproducibility ---
    mi_detail::Cache<T>::entropy_cache.clear();

    return blocks.empty() ? std::vector<int>() : blocks[0];
}


template<typename T>
std::vector<int> greedyEntropyMinVariableOrderSimd(const std::vector<std::vector<T>>& states) {
    using namespace mi_detail;
    if (states.empty()) return {};
    std::vector<int> variables;
    for (size_t i = 0; i < states[0].size(); ++i) variables.push_back(static_cast<int>(i));

    std::vector<std::vector<int>> blocks;
    for (int v : variables) blocks.push_back({v});

    while (blocks.size() > 1) {
        double bestEnt = 9999999.9;
        std::pair<int, int> bestPair{-1, -1};
        for (size_t i = 0; i < blocks.size(); ++i) {
            for (size_t j = i + 1; j < blocks.size(); ++j) {
                double ent = cached_entropy(states, concat(blocks[i], blocks[j]));
                if (ent < bestEnt) {
                    bestEnt = ent;
                    bestPair = {static_cast<int>(i), static_cast<int>(j)};
                }
            }
        }
        int i = bestPair.first, j = bestPair.second;
        std::vector<int> mergedBlock = concat(blocks[i], blocks[j]);
        std::vector<std::vector<int>> newBlocks;
        for (size_t k = 0; k < blocks.size(); ++k)
            if (static_cast<int>(k) != i && static_cast<int>(k) != j)
                newBlocks.push_back(blocks[k]);
        newBlocks.push_back(std::move(mergedBlock));
        blocks = std::move(newBlocks);
    }
    
    // --- Clear all caches for reproducibility ---
    mi_detail::Cache<T>::entropy_cache.clear();

    return blocks.empty() ? std::vector<int>() : blocks[0];
}


template<typename T>
std::vector<int> greedyEntropySingleMinVariableOrderSimd(const std::vector<std::vector<T>>& states) {
    using namespace mi_detail;
    if (states.empty()) return {};
    std::vector<int> variables;
    
    for (size_t i = 0; i < states[0].size(); ++i) variables.push_back(static_cast<int>(i));
    std::vector<std::pair<double, int>> ents;
    
    for (auto var : variables) {
        double ent = cached_entropy(states, {var});
        ents.emplace_back(std::pair<double, int>{ent, var});
    }

    std::stable_sort(ents.begin(), ents.end(), [](auto lhs, auto rhs) {
        return lhs.first <= rhs.first;
    });

    std::vector<int> order;
    for (auto ent_var : ents) {
        order.emplace_back(ent_var.second);
    }
    
    return order;
}

