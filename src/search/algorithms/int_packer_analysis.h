#ifndef ALGORITHMS_INT_PACKER_ANALYSIS_H
#define ALGORITHMS_INT_PACKER_ANALYSIS_H

#include <vector>
#include <cassert>
#include <ostream>
#include <unordered_set>
#include <algorithm>

class TaskProxy;

/*
  Pure functional implementation of the old greedy bin packing algorithm.
  This function takes a task proxy and returns a bin packing assignment.
*/
namespace int_packer_analysis {

// Structure to represent the packing assignment for a single variable
// This mirrors IntPacker::VariableInfo but is a simple data structure
struct VariableInfo {
    int range;       // Domain size of the variable
    int bin_index;   // Which bin this variable is assigned to
    int shift;       // Bit offset within the bin

    VariableInfo() : range(0), bin_index(-1), shift(0) {}
    VariableInfo(int range_val, int bin_idx, int shift_val)
        : range(range_val), bin_index(bin_idx), shift(shift_val) {}

    int get_bin_index() const {
        return bin_index;
    }

    int get_shift() const {
        return shift;
    }

    int get_range() const {
        return range;
    }
};

// Constants and helper functions
static const int BITS_PER_BIN = sizeof(unsigned int) * 8;

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

/*
  Compute the old greedy bin packing for the given task.

  This uses a simple greedy strategy: for each bin, repeatedly add the
  largest variable that still fits, until no more variables fit.

  Args:
    task_proxy: TaskProxy providing access to variable domain sizes.

  Returns:
    Vector of VariableInfo, one per variable, containing the bin assignment
    and bit offset for each variable. This is compatible with IntPacker::var_infos.
*/
static std::vector<VariableInfo> compute_greedy_bin_packing(const TaskProxy &task_proxy) {
    // Extract ranges from task_proxy
    std::vector<int> ranges;
    for (auto var : task_proxy.get_variables()) {
        ranges.push_back(var.get_domain_size());
    }

    int num_vars = ranges.size();
    std::vector<VariableInfo> var_infos(num_vars);

    // bits_to_vars[k] contains all variables that require exactly k
    // bits to encode. Once a variable is packed into a bin, it is
    // removed from this index.
    // Loop over the variables in reverse order to prefer variables with
    // low indices in case of ties. This might increase cache-locality.
    std::vector<std::vector<int>> bits_to_vars(BITS_PER_BIN + 1);
    for (int var = num_vars - 1; var >= 0; --var) {
        int bits = get_bit_size_for_range(ranges[var]);
        assert(bits <= BITS_PER_BIN);
        bits_to_vars[bits].push_back(var);
    }

    int num_bins = 0;
    int packed_vars = 0;
    while (packed_vars < num_vars) {
        // Pack one bin using greedy strategy
        ++num_bins;
        int bin_index = num_bins - 1;
        int used_bits = 0;

        while (true) {
            // Determine size of largest variable that still fits into the bin.
            int bits = BITS_PER_BIN - used_bits;
            while (bits > 0 && bits_to_vars[bits].empty())
                --bits;

            if (bits == 0) {
                // No more variables fit into the bin.
                // (This also happens when all variables have been packed.)
                break;
            }

            // We can pack another variable of size bits into the current bin.
            // Remove the variable from bits_to_vars and add it to the bin.
            std::vector<int> &best_fit_vars = bits_to_vars[bits];
            int var = best_fit_vars.back();
            best_fit_vars.pop_back();

            var_infos[var] = VariableInfo(ranges[var], bin_index, used_bits);
            used_bits += bits;
            ++packed_vars;
        }
    }

    return var_infos;
}

/*
  Analyze how many bins are touched by each operator's effects.

  For each operator, determines the unique set of bins that contain variables
  modified by any of the operator's effects. Reports average and maximum
  number of bins touched.

  This is a template function that works with any type that has a get_bin_index() method,
  including both int_packer_analysis::VariableInfo and IntPacker::VariableInfo.

  Args:
    task_proxy: TaskProxy providing access to operators and their effects.
    var_infos: Vector containing bin assignments for each variable.
    log: Output stream for logging the analysis results.
*/
template<typename VarInfoType>
static void analyze_operator_bin_touches(
    const TaskProxy &task_proxy,
    const std::vector<VarInfoType> &var_infos,
    std::ostream &log) {

    int total_bins_touched = 0;
    int max_bins_touched = 0;
    int num_operators = 0;

    // Calculate number of bins by finding the maximum bin index
    int num_bins = 0;
    for (const auto &var_info : var_infos) {
        int bin_idx = var_info.get_bin_index();
        if (bin_idx >= num_bins) {
            num_bins = bin_idx + 1;
        }
    }

    for (auto op : task_proxy.get_operators()) {
        std::unordered_set<int> touched_bins;

        // Iterate over all effects of this operator
        for (auto effect : op.get_effects()) {
            int var_id = effect.get_fact().get_variable().get_id();

            // Find which bin this variable is in
            int bin_index = var_infos[var_id].get_bin_index();
            touched_bins.insert(bin_index);
        }

        int bins_touched = touched_bins.size();
        total_bins_touched += bins_touched;
        max_bins_touched = std::max(max_bins_touched, bins_touched);
        ++num_operators;
    }

    double average_bins_touched = (num_operators > 0)
        ? static_cast<double>(total_bins_touched) / num_operators
        : 0.0;

    log << "Bin touch analysis for operators:" << std::endl;
    log << "  Number of bins: " << num_bins << std::endl;
    log << "  Average bins touched per operator: " << average_bins_touched << std::endl;
    log << "  Maximum bins touched by any operator: " << max_bins_touched << std::endl;
}

}

#endif

