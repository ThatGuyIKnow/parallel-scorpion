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

// Compute affinity between variables based on causal graph and operator co-occurrence
static vector<unordered_map<int, int>> compute_affinity(const TaskProxy &task_proxy, int num_vars) {
    vector<unordered_map<int, int>> affinity(num_vars);

    // Get causal graph for structural relationships
    const causal_graph::CausalGraph &cg = task_proxy.get_causal_graph();

    // Objective 1: Direct causal relationships have strong affinity
    // Variables that directly affect each other are accessed sequentially
    for (int v = 0; v < num_vars; ++v) {
        for (int u : cg.get_successors(v)) {
            affinity[v][u] += 5;  // Strong weight for direct causation
            affinity[u][v] += 5;
        }
    }

    // Objective 2: Operator-based affinity with smart weighting
    // Key insight: Smaller operators are executed more frequently in search,
    // and variables in the same small operator are accessed together more often
    for (OperatorProxy op : task_proxy.get_operators()) {
        unordered_set<int> precond_vars;
        unordered_set<int> effect_vars;
        unordered_set<int> all_vars;

        // Collect precondition variables (READ access)
        for (FactProxy pre : op.get_preconditions()) {
            int var = pre.get_variable().get_id();
            precond_vars.insert(var);
            all_vars.insert(var);
        }

        // Collect effect variables and their conditions (READ + WRITE access)
        for (EffectProxy eff : op.get_effects()) {
            int eff_var = eff.get_fact().get_variable().get_id();
            effect_vars.insert(eff_var);
            all_vars.insert(eff_var);

            // Effect condition variables (READ access)
            for (FactProxy cond : eff.get_conditions()) {
                int cond_var = cond.get_variable().get_id();
                precond_vars.insert(cond_var);
                all_vars.insert(cond_var);
            }
        }

        // Weight: Smaller operators (fewer variables) get higher weight
        // Because they're typically more frequently applicable
        int op_size = all_vars.size();
        int base_weight = max(1, 10 - op_size);  // Smaller ops get more weight

        // Strong affinity between variables that are BOTH READ (preconditions)
        // These are checked together during applicability testing
        for (int v : precond_vars) {
            for (int u : precond_vars) {
                if (v != u) {
                    affinity[v][u] += base_weight * 2;  // READ-READ: checked together
                }
            }
        }

        // Medium affinity between READ (precondition) and WRITE (effect) variables
        // These are accessed sequentially when applying the operator
        for (int v : precond_vars) {
            for (int u : effect_vars) {
                if (v != u) {
                    affinity[v][u] += base_weight;
                    affinity[u][v] += base_weight;
                }
            }
        }

        // Weaker affinity between variables that are both WRITTEN (effects)
        // These are updated together but not typically read together
        for (int v : effect_vars) {
            for (int u : effect_vars) {
                if (v != u) {
                    affinity[v][u] += base_weight / 2;
                }
            }
        }
    }

    // Objective 3: Axiom variables have strong affinity
    // Axioms are evaluated frequently, so their variables should be close
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

        // Higher weight for axiom variables - they're evaluated on every state
        for (int v : axiom_vars) {
            for (int u : axiom_vars) {
                if (v != u) {
                    affinity[v][u] += 8;  // Strong weight for axioms
                }
            }
        }
    }

    // Objective 4: Goal variables should be grouped together
    // Goals are checked frequently during search (heuristic computation, goal test)
    unordered_set<int> goal_vars;
    for (FactProxy goal : task_proxy.get_goals()) {
        goal_vars.insert(goal.get_variable().get_id());
    }
    for (int v : goal_vars) {
        for (int u : goal_vars) {
            if (v != u) {
                affinity[v][u] += 10;  // Very high weight for goal variables
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

    utils::g_log << "Starting affinity-based bin packing for " << num_vars << " variables" << endl;

    // Step 0: Compute bit sizes and affinity
    utils::g_log << "Computing bit sizes for all variables..." << endl;
    vector<int> bit_sizes(num_vars);
    for (int v = 0; v < num_vars; ++v) {
        bit_sizes[v] = get_bit_size_for_range(ranges[v]);
    }

    utils::g_log << "Computing variable affinity from task structure..." << endl;
    vector<unordered_map<int, int>> affinity = compute_affinity(task_proxy, num_vars);

    // Compute total affinity for each variable
    vector<int> total_affinity(num_vars, 0);
    for (int v = 0; v < num_vars; ++v) {
        for (const auto &pair : affinity[v]) {
            total_affinity[v] += pair.second;
        }
    }

    utils::g_log << "Affinity computation complete." << endl;

    // Debug: Print affinity statistics
    int total_affinity_edges = 0;
    int max_affinity_value = 0;
    for (int v = 0; v < num_vars; ++v) {
        for (const auto &pair : affinity[v]) {
            total_affinity_edges++;
            max_affinity_value = max(max_affinity_value, pair.second);
        }
    }
    utils::g_log << "Affinity statistics: " << total_affinity_edges << " edges, max value: " << max_affinity_value << endl;

    // Debug: Print bit size distribution
    unordered_map<int, int> bit_size_counts;
    for (int v = 0; v < num_vars; ++v) {
        bit_size_counts[bit_sizes[v]]++;
    }
    utils::g_log << "Bit size distribution: ";
    for (const auto &pair : bit_size_counts) {
        utils::g_log << pair.first << "-bit:" << pair.second << " vars; ";
    }
    utils::g_log << endl;

    // Sort variables by descending bit size (primary) and descending total affinity (secondary)
    utils::g_log << "Sorting variables by bit size and affinity..." << endl;
    vector<int> vars_sorted(num_vars);
    for (int v = 0; v < num_vars; ++v) {
        vars_sorted[v] = v;
    }

    sort(vars_sorted.begin(), vars_sorted.end(), [&](int v1, int v2) {
        if (bit_sizes[v1] != bit_sizes[v2]) {
            return bit_sizes[v1] > bit_sizes[v2]; // Descending bit size
        }
        return total_affinity[v1] > total_affinity[v2]; // Descending total affinity
    });

    // Step 1: Greedy bin packing with affinity tie-breaks
    utils::g_log << "Step 1: Greedy bin packing with affinity tie-breaks..." << endl;
    vector<BinInfo> bins;
    unordered_map<int, int> bin_of; // var -> bin index

    // For small variables: limit bin size to encourage better clustering
    const int MAX_SMALL_VARS_PER_BIN = 8;
    int affinity_decisions = 0;
    int fill_decisions = 0;

    for (int v : vars_sorted) {
        int best_bin_idx = -1;
        int best_fill = -1;
        int best_aff = numeric_limits<int>::min();

        for (size_t i = 0; i < bins.size(); ++i) {
            BinInfo &bin = bins[i];
            if (bin.used_bits + bit_sizes[v] > BITS_PER_BIN) {
                continue;
            }

            // For small variables: enforce max vars per bin to force better clustering
            if (bit_sizes[v] <= 2 && (int)bin.vars.size() >= MAX_SMALL_VARS_PER_BIN) {
                continue;
            }

            int remaining = BITS_PER_BIN - (bin.used_bits + bit_sizes[v]);
            int fill = BITS_PER_BIN - remaining; // total used after adding v

            // Compute affinity gain if we place v here
            int aff_gain = 0;
            for (int w : bin.vars) {
                aff_gain += affinity[v][w];
            }

            // Strategy depends on variable size:
            // For small variables (1-2 bits): prioritize AFFINITY over fill
            // Many small vars fit in one bin, so fill doesn't matter much
            // For larger variables (3+ bits): prioritize FILL over affinity
            // Fewer large vars fit, so packing efficiency matters more
            bool is_better = false;
            if (best_bin_idx == -1) {
                // First valid bin we've found
                is_better = true;
            } else if (bit_sizes[v] <= 2) {
                // Small variable: prioritize affinity, then fill
                if (aff_gain > best_aff) {
                    is_better = true;
                    affinity_decisions++;
                } else if (aff_gain == best_aff && fill > best_fill) {
                    is_better = true;
                    fill_decisions++;
                }
            } else {
                // Larger variable: prioritize fill, then affinity (original strategy)
                if (fill > best_fill) {
                    is_better = true;
                    fill_decisions++;
                } else if (fill == best_fill && aff_gain > best_aff) {
                    is_better = true;
                    affinity_decisions++;
                }
            }

            if (is_better) {
                best_bin_idx = i;
                best_fill = fill;
                best_aff = aff_gain;
            }
        }

        if (best_bin_idx != -1) {
            // Place v into best existing bin
            bins[best_bin_idx].vars.insert(v);
            bins[best_bin_idx].used_bits += bit_sizes[v];
            bin_of[v] = best_bin_idx;
        } else {
            // Open a new bin
            BinInfo new_bin;
            new_bin.vars.insert(v);
            new_bin.used_bits = bit_sizes[v];
            bin_of[v] = bins.size();
            bins.push_back(new_bin);
        }
    }

    utils::g_log << "Initial packing complete: " << bins.size() << " bins created" << endl;
    utils::g_log << "Decisions: " << affinity_decisions << " by affinity, "
                 << fill_decisions << " by fill" << endl;

    // Step 2: Optional local improvement pass
    utils::g_log << "Step 2: Local improvement pass (max 10 iterations)..." << endl;
    // Try moving variables to improve affinity without increasing bin count
    bool improved = true;
    int max_iterations = 10;
    int iteration = 0;

    while (improved && iteration < max_iterations) {
        improved = false;
        iteration++;

        for (int v = 0; v < num_vars; ++v) {
            int current_bin_idx = bin_of[v];
            BinInfo &current_bin = bins[current_bin_idx];

            // Compute current affinity contribution
            int current_affinity = 0;
            for (int w : current_bin.vars) {
                if (w != v) {
                    current_affinity += affinity[v][w];
                }
            }

            // Try moving to other bins
            int best_new_bin_idx = -1;
            int best_new_affinity = current_affinity;

            for (size_t i = 0; i < bins.size(); ++i) {
                if ((int)i == current_bin_idx) continue;

                BinInfo &other_bin = bins[i];

                // Check if v fits in other_bin after removing it from current bin
                int other_used_without_v = other_bin.used_bits;
                if (other_used_without_v + bit_sizes[v] > BITS_PER_BIN) {
                    continue;
                }

                // Compute new affinity if we move v to this bin
                int new_affinity = 0;
                for (int w : other_bin.vars) {
                    new_affinity += affinity[v][w];
                }

                if (new_affinity > best_new_affinity) {
                    best_new_bin_idx = i;
                    best_new_affinity = new_affinity;
                }
            }

            // Move if beneficial
            if (best_new_bin_idx != -1) {
                // Remove from current bin
                current_bin.vars.erase(v);
                current_bin.used_bits -= bit_sizes[v];

                // Add to new bin
                bins[best_new_bin_idx].vars.insert(v);
                bins[best_new_bin_idx].used_bits += bit_sizes[v];
                bin_of[v] = best_new_bin_idx;

                improved = true;
            }
        }
    }

    utils::g_log << "Local improvement converged after " << iteration << " iterations" << endl;

    // Remove empty bins
    utils::g_log << "Cleaning up empty bins..." << endl;
    vector<BinInfo> non_empty_bins;
    unordered_map<int, int> old_to_new_bin;
    for (size_t i = 0; i < bins.size(); ++i) {
        if (!bins[i].vars.empty()) {
            old_to_new_bin[i] = non_empty_bins.size();
            non_empty_bins.push_back(bins[i]);
        }
    }
    bins = non_empty_bins;

    // Update bin_of with new indices
    for (auto &pair : bin_of) {
        pair.second = old_to_new_bin[pair.second];
    }

    // Assign variables to bins and create VariableInfo
    utils::g_log << "Finalizing bin assignments..." << endl;
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

    utils::g_log << "Affinity-based packing complete: " << num_bins << " bins used" << endl;

    // Compute what the original greedy packing would have used for comparison
    vector<vector<int>> original_bin_assignment;
    int original_bins = compute_original_bin_count(bit_sizes, &original_bin_assignment);

    utils::g_log << "Original greedy packing would have used: " << original_bins << " bins" << endl;
    if (num_bins < original_bins) {
        utils::g_log << "Affinity-based packing saved " << (original_bins - num_bins) << " bins!" << endl;
    } else if (num_bins == original_bins) {
        utils::g_log << "Affinity-based packing used the same number of bins (but with better locality)" << endl;
    } else {
        utils::g_log << "Affinity-based packing used " << (num_bins - original_bins) << " more bins for improved locality" << endl;
    }

    // Print detailed bin assignments
    utils::g_log << "\n=== Original Greedy Packing ===" << endl;
    utils::g_log << "Original: ";
    for (size_t bin_idx = 0; bin_idx < original_bin_assignment.size(); ++bin_idx) {
        if (bin_idx > 0) utils::g_log << " | ";
        for (size_t i = 0; i < original_bin_assignment[bin_idx].size(); ++i) {
            utils::g_log << "(" << original_bin_assignment[bin_idx][i] << ")";
        }
    }
    utils::g_log << endl;

    utils::g_log << "\n=== Affinity-Based Packing ===" << endl;
    utils::g_log << "Affinity: ";
    for (size_t bin_idx = 0; bin_idx < bins.size(); ++bin_idx) {
        if (bin_idx > 0) utils::g_log << " | ";
        // Sort variables in bin for consistent display
        vector<int> vars_in_bin(bins[bin_idx].vars.begin(), bins[bin_idx].vars.end());
        sort(vars_in_bin.begin(), vars_in_bin.end(), [&](int v1, int v2) {
            return bit_sizes[v1] > bit_sizes[v2];
        });
        for (size_t i = 0; i < vars_in_bin.size(); ++i) {
            utils::g_log << "(" << vars_in_bin[i] << ")";
        }
    }
    utils::g_log << endl;

    // Check if the assignments are equal
    // Compare bin assignments by checking if each variable is in the same bin in both methods
    bool assignments_equal = (original_bin_assignment.size() == bins.size());
    if (assignments_equal) {
        // Create a mapping from variable to bin for the original assignment
        vector<int> original_var_to_bin(num_vars);
        for (size_t bin_idx = 0; bin_idx < original_bin_assignment.size(); ++bin_idx) {
            for (int var : original_bin_assignment[bin_idx]) {
                original_var_to_bin[var] = bin_idx;
            }
        }

        // Check if each variable is in the same bin (after accounting for potential bin reordering)
        // Two assignments are equal if the grouping is the same, even if bin numbers differ
        // So we check if variables that are together in one method are together in the other
        for (int v1 = 0; v1 < num_vars && assignments_equal; ++v1) {
            for (int v2 = v1 + 1; v2 < num_vars; ++v2) {
                bool together_in_original = (original_var_to_bin[v1] == original_var_to_bin[v2]);
                bool together_in_affinity = (bin_of[v1] == bin_of[v2]);
                if (together_in_original != together_in_affinity) {
                    assignments_equal = false;
                    break;
                }
            }
        }
    }

    utils::g_log << "\nAssignments are " << (assignments_equal ? "EQUAL" : "DIFFERENT") << endl;
    utils::g_log << "========================================\n" << endl;
}

}
