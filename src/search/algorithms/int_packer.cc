#include "int_packer.h"

#include "../task_proxy.h"
#include "../task_utils/causal_graph.h"
#include "../utils/logging.h"

#include <cassert>
#include <algorithm>
#include <unordered_map>
#include <unordered_set>
#include <limits>
#include <set>

using namespace std;

namespace int_packer {
static const int BITS_PER_BIN = sizeof(IntPacker::Bin) * 8;

static IntPacker::Bin get_bit_mask(int from, int to) {
    // Return mask with all bits in the range [from, to) set to 1.
    assert(from >= 0 && to >= from && to <= BITS_PER_BIN);
    int length = to - from;
    if (length == BITS_PER_BIN) {
        // 1U << BITS_PER_BIN has undefined behaviour in C++; e.g.
        // 1U << 32 == 1 (not 0) on 32-bit Intel platforms. Hence this
        // special case.
        assert(from == 0 && to == BITS_PER_BIN);
        return ~IntPacker::Bin(0);
    } else {
        return ((IntPacker::Bin(1) << length) - 1) << from;
    }
}

static int get_bit_size_for_range(int range) {
    assert(range >= 1);
    // Domains in domain-abstracted tasks may have size one.
    if (range == 1) {
        return 1;
    }
    int num_bits = 0;
    while ((1U << num_bits) < static_cast<unsigned int>(range))
        ++num_bits;
    return num_bits;
}

class IntPacker::VariableInfo {
    int range;
    int bin_index;
    int shift;
    Bin read_mask;
    Bin clear_mask;
public:
    VariableInfo(int range_, int bin_index_, int shift_)
        : range(range_),
          bin_index(bin_index_),
          shift(shift_) {
        int bit_size = get_bit_size_for_range(range);
        read_mask = get_bit_mask(shift, shift + bit_size);
        clear_mask = ~read_mask;
    }

    VariableInfo()
        : bin_index(-1), shift(0), read_mask(0), clear_mask(0) {
        // Default constructor needed for resize() in pack_bins.
    }

    ~VariableInfo() {
    }

    int get(const Bin *buffer) const {
        return (buffer[bin_index] & read_mask) >> shift;
    }

    void set(Bin *buffer, int value) const {
        assert(value >= 0 && value < range);
        Bin &bin = buffer[bin_index];
        bin = (bin & clear_mask) | (value << shift);
    }
};


IntPacker::IntPacker(const vector<int> &ranges)
    : num_bins(0), task(nullptr) {
    pack_bins(ranges);
}

IntPacker::IntPacker(const TaskProxy &task_proxy, const vector<int> &ranges)
    : num_bins(0), task(&task_proxy.get_task()) {
    pack_bins(ranges);
}

IntPacker::~IntPacker() {
}

int IntPacker::get(const Bin *buffer, int var) const {
    return var_infos[var].get(buffer);
}

void IntPacker::set(Bin *buffer, int var, int value) const {
    var_infos[var].set(buffer, value);
}

TaskProxy IntPacker::get_task_proxy() const {
    assert(task);
    return TaskProxy(*task);
}

// Compute affinity between variables based on operator co-occurrence with probability weighting
static vector<unordered_map<int, double>> compute_affinity(const TaskProxy &task_proxy, int num_vars) {
    vector<unordered_map<int, double>> affinity(num_vars);

    // Compute operator probabilities p[a] = product(1/D[v]) for v in vars(a)
    for (OperatorProxy op : task_proxy.get_operators()) {
        unordered_set<int> op_vars;

        // Collect all variables in the operator (preconditions and effects)
        for (FactProxy pre : op.get_preconditions()) {
            op_vars.insert(pre.get_variable().get_id());
        }

        for (EffectProxy eff : op.get_effects()) {
            op_vars.insert(eff.get_fact().get_variable().get_id());

            // Effect condition variables
            for (FactProxy cond : eff.get_conditions()) {
                op_vars.insert(cond.get_variable().get_id());
            }
        }

        // Compute p[a] = product of 1/D[v] for all variables in operator
        double p_a = 1.0;
        for (int v : op_vars) {
            int domain_size = task_proxy.get_variables()[v].get_domain_size();
            p_a *= (1.0 / domain_size);
        }

        // Add pairwise affinities for all pairs of variables in this operator
        vector<int> vars_list(op_vars.begin(), op_vars.end());
        for (size_t i = 0; i < vars_list.size(); ++i) {
            for (size_t j = i + 1; j < vars_list.size(); ++j) {
                int u = vars_list[i];
                int v = vars_list[j];
                affinity[u][v] += p_a;
                affinity[v][u] += p_a;
            }
        }
    }

    // Also process axioms (they tend to be evaluated frequently)
    for (OperatorProxy axiom : task_proxy.get_axioms()) {
        unordered_set<int> axiom_vars;

        for (FactProxy pre : axiom.get_preconditions()) {
            axiom_vars.insert(pre.get_variable().get_id());
        }

        for (EffectProxy eff : axiom.get_effects()) {
            axiom_vars.insert(eff.get_fact().get_variable().get_id());
            for (FactProxy cond : eff.get_conditions()) {
                axiom_vars.insert(cond.get_variable().get_id());
            }
        }

        // Compute p[axiom] similarly
        double p_axiom = 1.0;
        for (int v : axiom_vars) {
            int domain_size = task_proxy.get_variables()[v].get_domain_size();
            p_axiom *= (1.0 / domain_size);
        }

        // Add pairwise affinities weighted by p_axiom * 2 (axioms evaluated more often)
        vector<int> vars_list(axiom_vars.begin(), axiom_vars.end());
        for (size_t i = 0; i < vars_list.size(); ++i) {
            for (size_t j = i + 1; j < vars_list.size(); ++j) {
                int u = vars_list[i];
                int v = vars_list[j];
                affinity[u][v] += p_axiom;
                affinity[v][u] += p_axiom;
            }
        }
    }

    return affinity;
}

// Structure to represent a bin during packing
struct BinInfo {
    unordered_set<int> vars;
    int used_bits;

    BinInfo() : used_bits(0) {}
};

// Compute how many bins the original greedy algorithm would use
static int compute_original_bin_count(const vector<int> &bit_sizes, vector<vector<int>> *original_bin_assignment = nullptr) {
    int num_vars = bit_sizes.size();

    // Sort by descending bit size (original algorithm behavior)
    vector<int> original_vars_sorted(num_vars);
    for (int v = 0; v < num_vars; ++v) {
        original_vars_sorted[v] = v;
    }
    sort(original_vars_sorted.begin(), original_vars_sorted.end(), [&](int v1, int v2) {
        return bit_sizes[v1] > bit_sizes[v2];
    });

    // Simulate original greedy bin packing (First-Fit Decreasing)
    int current_bin_bits = 0;
    int original_bins = 1;
    vector<vector<int>> bins_assignment;
    bins_assignment.push_back(vector<int>());

    for (int v : original_vars_sorted) {
        if (current_bin_bits + bit_sizes[v] > BITS_PER_BIN) {
            // Need a new bin
            original_bins++;
            current_bin_bits = bit_sizes[v];
            bins_assignment.push_back(vector<int>());
            bins_assignment.back().push_back(v);
        } else {
            current_bin_bits += bit_sizes[v];
            bins_assignment.back().push_back(v);
        }
    }

    if (original_bin_assignment) {
        *original_bin_assignment = bins_assignment;
    }

    return original_bins;
}

void IntPacker::pack_bins(const vector<int> &ranges) {
    assert(var_infos.empty());

    int num_vars = ranges.size();
    var_infos.resize(num_vars);

    // If no task available, fall back to simple greedy packing
    if (!task) {
        pack_bins_simple(ranges);
        return;
    }

    // Use affinity-based packing
    TaskProxy task_proxy(*task);
    pack_bins_affinity(task_proxy, ranges);
}

// Simple greedy packing (original algorithm)
void IntPacker::pack_bins_simple(const vector<int> &ranges) {
    int num_vars = ranges.size();

    vector<vector<int>> bits_to_vars(BITS_PER_BIN + 1);
    for (int var = num_vars - 1; var >= 0; --var) {
        int bits = get_bit_size_for_range(ranges[var]);
        assert(bits <= BITS_PER_BIN);
        bits_to_vars[bits].push_back(var);
    }

    int packed_vars = 0;
    while (packed_vars != num_vars)
        packed_vars += pack_one_bin(ranges, bits_to_vars);
}

int IntPacker::pack_one_bin(const vector<int> &ranges,
                            vector<vector<int>> &bits_to_vars) {
    // Returns the number of variables added to the bin. We pack each
    // bin with a greedy strategy, always adding the largest variable
    // that still fits.

    ++num_bins;
    int bin_index = num_bins - 1;
    int used_bits = 0;
    int num_vars_in_bin = 0;

    while (true) {
        // Determine size of largest variable that still fits into the bin.
        int bits = BITS_PER_BIN - used_bits;
        while (bits > 0 && bits_to_vars[bits].empty())
            --bits;

        if (bits == 0) {
            // No more variables fit into the bin.
            // (This also happens when all variables have been packed.)
            return num_vars_in_bin;
        }

        // We can pack another variable of size bits into the current bin.
        // Remove the variable from bits_to_vars and add it to the bin.
        vector<int> &best_fit_vars = bits_to_vars[bits];
        int var = best_fit_vars.back();
        best_fit_vars.pop_back();

        var_infos[var] = VariableInfo(ranges[var], bin_index, used_bits);
        used_bits += bits;
        ++num_vars_in_bin;
    }
}

// Affinity-based packing
void IntPacker::pack_bins_affinity(const TaskProxy &task_proxy, const vector<int> &ranges) {
    int num_vars = ranges.size();

    // Step 0: Compute bit sizes
    vector<int> bit_sizes(num_vars);
    for (int v = 0; v < num_vars; ++v) {
        bit_sizes[v] = get_bit_size_for_range(ranges[v]);
    }

    // Step 1: Compute pairwise affinities
    vector<unordered_map<int, double>> affinity = compute_affinity(task_proxy, num_vars);

    // Step 2: Compute priority score per variable: total affinity per-bit
    vector<double> total_affinity(num_vars, 0.0);
    vector<double> priority_score(num_vars, 0.0);

    for (int v = 0; v < num_vars; ++v) {
        for (const auto &pair : affinity[v]) {
            total_affinity[v] += pair.second;
        }
        // priority_score = total_affinity / bits (affinity density)
        if (bit_sizes[v] > 0) {
            priority_score[v] = total_affinity[v] / bit_sizes[v];
        }
    }

    // Step 3: Sort variables by descending priority_score
    vector<int> vars_sorted(num_vars);
    for (int v = 0; v < num_vars; ++v) {
        vars_sorted[v] = v;
    }

    sort(vars_sorted.begin(), vars_sorted.end(), [&](int v1, int v2) {
        return priority_score[v1] > priority_score[v2];
    });

    // Greedy affinity-aware first-fit by descending priority
    vector<BinInfo> bins;
    unordered_map<int, int> bin_of; // var -> bin index

    for (int v : vars_sorted) {
        int best_bin_idx = -1;
        double best_gain = -numeric_limits<double>::infinity();
        int best_remaining = BITS_PER_BIN;

        for (size_t i = 0; i < bins.size(); ++i) {
            BinInfo &bin = bins[i];
            if (bin.used_bits + bit_sizes[v] > BITS_PER_BIN) {
                continue;
            }

            // gain = sum affinity between v and vars already in bin
            double gain = 0.0;
            for (int w : bin.vars) {
                auto it = affinity[v].find(w);
                if (it != affinity[v].end()) {
                    gain += it->second;
                }
            }

            // normalized gain per-bit
            double norm_gain = (bit_sizes[v] > 0) ? (gain / bit_sizes[v]) : 0.0;
            int remaining = BITS_PER_BIN - (bin.used_bits + bit_sizes[v]);

            // Choose bin with maximum norm_gain (tie-break by smaller remaining bits)
            if (norm_gain > best_gain ||
                (norm_gain == best_gain && remaining < best_remaining)) {
                best_gain = norm_gain;
                best_bin_idx = i;
                best_remaining = remaining;
            }
        }

        if (best_bin_idx != -1) {
            // Assign v to best_bin
            bins[best_bin_idx].vars.insert(v);
            bins[best_bin_idx].used_bits += bit_sizes[v];
            bin_of[v] = best_bin_idx;
        } else {
            // Open new bin
            BinInfo new_bin;
            new_bin.vars.insert(v);
            new_bin.used_bits = bit_sizes[v];
            bin_of[v] = bins.size();
            bins.push_back(new_bin);
        }
    }

    // Assign variables to bins and create VariableInfo
    num_bins = bins.size();

    for (size_t bin_idx = 0; bin_idx < bins.size(); ++bin_idx) {
        int shift = 0;
        // Sort variables in each bin by descending bit size for consistent packing
        vector<int> vars_in_bin(bins[bin_idx].vars.begin(), bins[bin_idx].vars.end());
        sort(vars_in_bin.begin(), vars_in_bin.end(), [&](int v1, int v2) {
            return bit_sizes[v1] > bit_sizes[v2];
        });

        for (int v : vars_in_bin) {
            var_infos[v] = VariableInfo(ranges[v], bin_idx, shift);
            shift += bit_sizes[v];
        }
    }

    // Compare with original greedy packing
    vector<vector<int>> original_bin_assignment;
    int original_bins = compute_original_bin_count(bit_sizes, &original_bin_assignment);

    // Check if the packings are identical
    bool packings_identical = (num_bins == original_bins);
    if (packings_identical) {
        // Convert affinity-based bins to sorted vectors for comparison
        vector<std::set<int>> affinity_bins_sorted;
        for (const BinInfo &bin : bins) {
            affinity_bins_sorted.push_back(std::set<int>(bin.vars.begin(), bin.vars.end()));
        }

        vector<std::set<int>> original_bins_sorted;
        for (const vector<int> &bin : original_bin_assignment) {
            original_bins_sorted.push_back(std::set<int>(bin.begin(), bin.end()));
        }

        // Sort both bin collections to compare (order of bins doesn't matter)
        sort(affinity_bins_sorted.begin(), affinity_bins_sorted.end());
        sort(original_bins_sorted.begin(), original_bins_sorted.end());

        packings_identical = (affinity_bins_sorted == original_bins_sorted);
    }

    // Compute total intra-bin affinity for affinity-based packing
    double affinity_packing_score = 0.0;
    for (const BinInfo &bin : bins) {
        for (int v : bin.vars) {
            for (int w : bin.vars) {
                if (v < w) { // count each pair once
                    auto it = affinity[v].find(w);
                    if (it != affinity[v].end()) {
                        affinity_packing_score += it->second;
                    }
                }
            }
        }
    }

    // Compute total intra-bin affinity for original packing
    double original_packing_score = 0.0;
    for (const vector<int> &bin : original_bin_assignment) {
        for (size_t i = 0; i < bin.size(); ++i) {
            for (size_t j = i + 1; j < bin.size(); ++j) {
                int v = bin[i];
                int w = bin[j];
                auto it = affinity[v].find(w);
                if (it != affinity[v].end()) {
                    original_packing_score += it->second;
                }
            }
        }
    }

    // Log comparison results
    utils::g_log << "Bin packing comparison:" << endl;
    utils::g_log << "  Original (greedy):  " << original_bins << " bins, "
                 << "affinity score: " << original_packing_score << endl;
    utils::g_log << "  Affinity-based:     " << num_bins << " bins, "
                 << "affinity score: " << affinity_packing_score << endl;

    if (packings_identical) {
        utils::g_log << "  Packings are IDENTICAL" << endl;
    } else if (num_bins < original_bins) {
        utils::g_log << "  Affinity-based packing uses "
                     << (original_bins - num_bins) << " fewer bins";
        if (affinity_packing_score > original_packing_score) {
            utils::g_log << " and has "
                         << ((affinity_packing_score - original_packing_score) / original_packing_score * 100)
                         << "% higher affinity score" << endl;
        } else {
            utils::g_log << endl;
        }
    } else if (num_bins == original_bins) {
        utils::g_log << "  Both packings use the same number of bins but DIFFERENT variable assignments";
        if (affinity_packing_score > original_packing_score) {
            utils::g_log << ", and affinity-based has "
                         << ((affinity_packing_score - original_packing_score) / original_packing_score * 100)
                         << "% higher affinity score" << endl;
        } else if (affinity_packing_score < original_packing_score) {
            utils::g_log << ", and affinity-based has "
                         << ((original_packing_score - affinity_packing_score) / affinity_packing_score * 100)
                         << "% lower affinity score" << endl;
        } else {
            utils::g_log << " with equal affinity scores" << endl;
        }
    } else {
        utils::g_log << "  Affinity-based packing uses "
                     << (num_bins - original_bins) << " more bins";
        if (affinity_packing_score > original_packing_score) {
            utils::g_log << " but has "
                         << ((affinity_packing_score - original_packing_score) / original_packing_score * 100)
                         << "% higher affinity score" << endl;
        } else {
            utils::g_log << endl;
        }
    }
}

}
