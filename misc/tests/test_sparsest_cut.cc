// Standalone unit test for the pure sparsest-cut bisection used by GRAZHDA*.
//
// Validates the implementation against the objective from Jinnai & Fukunaga,
// "A Graph-Partitioning Based Approach for Parallel Best-First Search"
// (HSDIP-17): partition a domain transition graph into two abstract features
// maximizing sparsity = (|S1|*|S2|) / (sum of cut-edge weights)  [Eq. 5, k=2],
// with edge weights = number of ground actions inducing the transition.
//
// Compile & run (no Fast Downward dependencies):
//   g++ -std=c++20 -I src/search misc/tests/test_sparsest_cut.cc -o /tmp/tsc && /tmp/tsc
#include "parallel_hash/sparsest_cut.h"

#include <cassert>
#include <cmath>
#include <iostream>
#include <vector>

using namespace distribution_hash;

// Count cut edges and their total weight for a given bipartition.
static std::pair<int, double> cut_stats(
        const std::vector<bool> &side, const std::vector<WeightedEdge> &edges) {
    int count = 0;
    double weight = 0.0;
    for (const WeightedEdge &e : edges) {
        if (side[e.u] != side[e.v]) {
            ++count;
            weight += e.weight;
        }
    }
    return {count, weight};
}

static int side_size(const std::vector<bool> &side, bool which) {
    int n = 0;
    for (bool b : side) n += (b == which);
    return n;
}

static int passed = 0;
#define CHECK(cond) do { if (!(cond)) { \
    std::cerr << "FAIL: " #cond " (line " << __LINE__ << ")\n"; return false; } \
    ++passed; } while (0)

// 1. Path 0-1-2-3: the sparsest, most balanced cut is the middle edge (1,2):
//    |S1|=|S2|=2, 1 cut edge -> sparsity 4. Beats end cuts (sparsity 3) and the
//    interleaved cut {0,2}|{1,3} (3 cut edges, sparsity 1.33).
static bool test_path() {
    std::vector<WeightedEdge> edges = {{0,1,1},{1,2,1},{2,3,1}};
    auto side = sparsest_cut_bisection(4, edges);
    auto [count, weight] = cut_stats(side, edges);
    CHECK(count == 1);
    CHECK(side_size(side, false) == 2 && side_size(side, true) == 2);
    CHECK(side[0] == side[1] && side[2] == side[3] && side[1] != side[2]);
    return true;
}

// 2. Barbell = two triangles {0,1,2} and {3,4,5} joined by a single bridge
//    (2,3). This is the essence of the paper's logistics package-location DTG:
//    the sparsest cut is the bridge (1 edge), giving sparsity 3*3/1 = 9, whereas
//    a greedy connectivity bisection (GreedyAFG) cuts 2 edges. Validates that we
//    cut the bridge, not a triangle edge.
static bool test_barbell_bridge() {
    std::vector<WeightedEdge> edges = {
        {0,1,1},{1,2,1},{0,2,1},        // triangle A
        {3,4,1},{4,5,1},{3,5,1},        // triangle B
        {2,3,1}};                        // bridge
    auto side = sparsest_cut_bisection(6, edges);
    auto [count, weight] = cut_stats(side, edges);
    CHECK(count == 1);                   // only the bridge is cut (paper: 1, not 2)
    CHECK(side[0]==side[1] && side[1]==side[2]);   // triangle A together
    CHECK(side[3]==side[4] && side[4]==side[5]);   // triangle B together
    CHECK(side[2] != side[3]);                     // split across the bridge
    return true;
}

// 3. Disconnected components -> cut weight 0 (maximal sparsity); must split
//    along the components.
static bool test_disconnected() {
    std::vector<WeightedEdge> edges = {{0,1,1},{2,3,1}};   // no edge between pairs
    auto side = sparsest_cut_bisection(4, edges);
    auto [count, weight] = cut_stats(side, edges);
    CHECK(count == 0);
    CHECK(side[0]==side[1] && side[2]==side[3] && side[1]!=side[2]);
    return true;
}

// 4. Edge weights matter: cut the light edge, never the heavy ones. Path with
//    weights 10-1-10; best cut is the middle (weight 1): sparsity 4/1 = 4.
static bool test_weighted() {
    std::vector<WeightedEdge> edges = {{0,1,10},{1,2,1},{2,3,10}};
    auto side = sparsest_cut_bisection(4, edges);
    auto [count, weight] = cut_stats(side, edges);
    CHECK(count == 1);
    CHECK(weight == 1.0);                // the light edge, not a heavy one
    return true;
}

// 5. Sparsity formula sanity on K4: every balanced 2-2 cut severs 4 edges,
//    sparsity = 2*2/4 = 1; the function must still return a valid non-empty
//    bipartition.
static bool test_clique() {
    std::vector<WeightedEdge> edges = {{0,1,1},{0,2,1},{0,3,1},{1,2,1},{1,3,1},{2,3,1}};
    auto side = sparsest_cut_bisection(4, edges);
    CHECK(side_size(side, false) >= 1 && side_size(side, true) >= 1);
    return true;
}

int main() {
    bool ok = true;
    ok &= test_path();
    ok &= test_barbell_bridge();
    ok &= test_disconnected();
    ok &= test_weighted();
    ok &= test_clique();
    if (ok) { std::cout << "ALL SPARSEST-CUT TESTS PASSED (" << passed << " checks)\n"; return 0; }
    std::cerr << "SPARSEST-CUT TESTS FAILED\n";
    return 1;
}
