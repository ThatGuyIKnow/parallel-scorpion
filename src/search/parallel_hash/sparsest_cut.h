#ifndef PARALLEL_HASH_SPARSEST_CUT_H
#define PARALLEL_HASH_SPARSEST_CUT_H

#include <algorithm>
#include <vector>

namespace distribution_hash {

// An undirected, weighted edge of a domain transition graph. `weight` is the
// number of ground actions inducing the transition (GRAZHDA* edge weight).
struct WeightedEdge {
    int u;
    int v;
    double weight;
};

// Partition the nodes {0..n-1} into two non-empty sets maximizing the sparsity
//     sparsity = (|S1| * |S2|) / (total weight of cut edges)
// from Jinnai & Fukunaga (sparsest cut, Eq. 5 with k=2). A disconnected cut
// (cut weight 0) has unbounded sparsity and dominates any positive-weight cut;
// among disconnected cuts we prefer the most balanced one. Returns a bool vector
// `side` of length n with side[i] = partition of node i (false = S1, true = S2).
// Node 0 is fixed to S1 to remove the S1<->S2 symmetry.
//
// Domain transition graphs are small, so we enumerate all bisections (2^(n-1)),
// exactly as the paper does. For unexpectedly large graphs we fall back to a
// balanced split to stay tractable.
inline std::vector<bool> sparsest_cut_bisection(
        int n, const std::vector<WeightedEdge> &edges) {
    if (n <= 1) {
        return std::vector<bool>(std::max(n, 0), false);
    }

    const int MAX_EXHAUSTIVE = 20;
    if (n > MAX_EXHAUSTIVE) {
        std::vector<bool> side(n, false);
        for (int i = n / 2; i < n; ++i) {
            side[i] = true;
        }
        return side;
    }

    std::vector<bool> best;
    bool best_disconnected = false;
    double best_value = -1.0;

    const int rest = n - 1; // nodes 1..n-1; node 0 is pinned to S1.
    for (unsigned long mask = 0; mask < (1UL << rest); ++mask) {
        std::vector<bool> side(n, false);
        int s2 = 0;
        for (int i = 0; i < rest; ++i) {
            if (mask & (1UL << i)) {
                side[i + 1] = true;
                ++s2;
            }
        }
        const int s1 = n - s2;
        if (s1 == 0 || s2 == 0) {
            continue; // both partitions must be non-empty
        }

        double cut = 0.0;
        for (const WeightedEdge &e : edges) {
            if (side[e.u] != side[e.v]) {
                cut += e.weight;
            }
        }

        const bool disconnected = (cut == 0.0);
        const double value = disconnected
            ? static_cast<double>(std::min(s1, s2)) // balance for disconnected cuts
            : (static_cast<double>(s1) * s2) / cut;  // sparsity (Eq. 5)

        const bool better = (disconnected != best_disconnected)
            ? disconnected            // a disconnected cut always wins
            : (value > best_value);
        if (better) {
            best_disconnected = disconnected;
            best_value = value;
            best = side;
        }
    }
    return best;
}

}

#endif
