#include "int_packer.h"

#include "../task_proxy.h"
#include "../task_utils/causal_graph.h"
#include "../utils/logging.h"

#include <gtl/phmap.hpp>

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
    int range = 0;
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

    int get_range() const {
        return range;
    }

    int get_bin_index() const {
        return bin_index;
    }
};


IntPacker::IntPacker(const vector<int> &ranges)
    : num_bins(0), task(nullptr), debug(true) {
    pack_bins(ranges);
}

IntPacker::IntPacker(const TaskProxy &task_proxy, const vector<int> &ranges)
    : num_bins(0), task(&task_proxy.get_task()), debug(true) {
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

int IntPacker::get_bin_index(int var_id) const {
    return var_infos[var_id].get_bin_index();
}

int IntPacker::get_range(int var_id) const {
    return var_infos[var_id].get_range();
}

TaskProxy IntPacker::get_task_proxy() const {
    assert(task);
    return TaskProxy(*task);
}


static auto compute_affinity(const TaskProxy &task_proxy, int num_vars) {
    auto affinity = Affinity();

    for (const auto op : task_proxy.get_operators()) {
        for (const auto eff_lhs : op.get_effects()) {
            for (const auto eff_rhs : op.get_effects()) {
                int var_lhs = eff_lhs.get_fact().get_variable().get_id();
                int var_rhs = eff_rhs.get_fact().get_variable().get_id();
                if (var_lhs == var_rhs) continue;

                affinity[var_lhs][var_rhs] += 1;
            }
        }
    }

    return affinity;
}

static auto compute_degree(const Affinity& affinity, int var)
{
    int deg = 0;
    auto it = affinity.find(var);
    if (it != affinity.end()) {
        for (const auto& pair : it->second) {
            deg += pair.second;
        }
    }
    return deg;
}

// Structure to represent a bin during packing
struct BinInfo {
    unordered_set<int> vars;
    int used_bits;

    BinInfo() : used_bits(0) {}
};


void IntPacker::pack_bins(const vector<int> &ranges) {
    assert(var_infos.empty());

    int num_vars = ranges.size();
    var_infos.resize(num_vars);

    const auto task_proxy = TaskProxy(*task);
    // bits_to_vars[k] contains all variables that require exactly k
    // bits to encode. Once a variable is packed into a bin, it is
    // removed from this index.
    // Loop over the variables in reverse order to prefer variables with
    // low indices in case of ties. This might increase cache-locality.
    vector<vector<int>> bits_to_vars(BITS_PER_BIN + 1);
    auto unpacked_vars = gtl::flat_hash_set<int>();
    for (const auto& var : task_proxy.get_variables()) {
        int bits = get_bit_size_for_range(ranges[var.get_id()]);
        assert(bits <= BITS_PER_BIN);
        bits_to_vars[bits].push_back(var.get_id());
        unpacked_vars.insert(var.get_id());
    }


    // Step 1: Compute pairwise affinities
    auto affinity = compute_affinity(task_proxy, num_vars);

    auto bin_vars = vector<int>();
    auto fit_unpacked_vars = gtl::flat_hash_set<int>();
    while (!unpacked_vars.empty()) {
        bool is_even = num_bins % 2 == 0;
        if (is_even) {
            pack_one_bin(task_proxy, affinity, unpacked_vars, ranges, bits_to_vars, bin_vars, fit_unpacked_vars);
        } else {
            pack_one_bin(task_proxy, affinity, unpacked_vars, ranges, bits_to_vars, bin_vars, fit_unpacked_vars);
            bin_vars.clear();
        }
    }
}


int IntPacker::pack_one_bin(const TaskProxy& task,
                            const Affinity& affinity,
                            gtl::flat_hash_set<int>& unpacked_vars,
                            const vector<int> &ranges,
                            vector<vector<int>> &bits_to_vars,
                            std::vector<int>& bin_vars,
                            gtl::flat_hash_set<int>& fit_unpacked_vars)
    {
    if (debug)
        std::cout << "Packing bin [" << num_bins << "]" << std::endl;

    ++num_bins;
    int bin_index = num_bins - 1;
    int used_bits = 0;
    int num_vars_in_bin = 0;

    // Determine variables that fit into
    auto determine_unpacked_fit_vars = [](int available_bits, const vector<int> &ranges, const gtl::flat_hash_set<int>& unpacked_vars, gtl::flat_hash_set<int>& fit_unpacked_vars) {
        fit_unpacked_vars.clear();
        {
            for (const auto& var : unpacked_vars) {
                if (get_bit_size_for_range(ranges[var]) <= available_bits) {
                    fit_unpacked_vars.insert(var);
                }
            }
        }
    };

    determine_unpacked_fit_vars(BITS_PER_BIN - used_bits, ranges, unpacked_vars, fit_unpacked_vars);

    // If there are no variable for consideration, seed a new variable
    if (bin_vars.empty()) {
        const int seed = *std::max_element(fit_unpacked_vars.begin(), fit_unpacked_vars.end(),
                                     [&](int var_lhs, int var_rhs) {
                                        // First sort: highest range that fits wins
                                        if (ranges[var_lhs] == ranges[var_rhs]) {
                                            const auto lhs_derived = task.get_variables()[var_lhs].is_derived();
                                            const auto rhs_derived = task.get_variables()[var_rhs].is_derived();
                                            // Second sort: fluent wins over derived
                                            if (lhs_derived == rhs_derived) {
                                                const auto degree_lhs = compute_degree(affinity, var_lhs);
                                                const auto degree_rhs = compute_degree(affinity, var_rhs);
                                                // Third sort: highest degree wins
                                                return degree_lhs < degree_rhs;
                                            }
                                            return lhs_derived > rhs_derived;
                                        }
                                        return ranges[var_lhs] < ranges[var_rhs];
                                     });

        var_infos[seed] = VariableInfo(ranges[seed], bin_index, used_bits);
        used_bits += get_bit_size_for_range(ranges[seed]);
        auto& bit_vars = bits_to_vars[get_bit_size_for_range(ranges[seed])];
        bit_vars.erase(std::find(bit_vars.begin(), bit_vars.end(), seed));
        ++num_vars_in_bin;
        bin_vars.push_back(seed);
        unpacked_vars.erase(seed);
        fit_unpacked_vars.erase(seed);

        if (debug) {
            std::cout << "Choosing seed [" << seed << "] derived? [" << task.get_variables()[seed].is_derived() << "] with domain size [" << ranges[seed] << "] with degree [" << compute_degree(affinity, seed) << "]" << std::endl;
        }
    }

    auto gain = std::vector<int>(affinity.size(), 0);
    while (true)
    {
        // Determine size of largest variable that still fits into the bin.

        determine_unpacked_fit_vars(BITS_PER_BIN - used_bits, ranges, unpacked_vars, fit_unpacked_vars);
        if (fit_unpacked_vars.empty()) {
            return num_vars_in_bin;
        }

        // Compute gain of adding variable into current bin
        std::fill(gain.begin(), gain.end(), 0);
        for (int var : fit_unpacked_vars)
        {
            auto it_var = affinity.find(var);
            if (it_var != affinity.end()) {
                for (int bin_var : bin_vars)
                {
                    auto it_bin = it_var->second.find(bin_var);
                    if (it_bin != it_var->second.end()) {
                        gain[var] += it_bin->second;
                    }
                }
            }
        }

        const auto next_var = *std::max_element(fit_unpacked_vars.begin(), fit_unpacked_vars.end(),
            [&](int var_lhs, int var_rhs) {
                
                // First sort: highest range that fits wins
                if (ranges[var_lhs] == ranges[var_rhs]) {                
                    // Second sort: fluent wins over derived
                    const auto lhs_derived = task.get_variables()[var_lhs].is_derived();
                    const auto rhs_derived = task.get_variables()[var_rhs].is_derived();
                    if (lhs_derived == rhs_derived) {
                        // Third sort: highest gain wins
                        if (gain[var_lhs] == gain[var_rhs]) {
                            const auto degree_lhs = compute_degree(affinity, var_lhs);
                            const auto degree_rhs = compute_degree(affinity, var_rhs);
                            // Fourth sort: highest degree wins
                            return degree_lhs < degree_rhs;
                        }
                        return gain[var_lhs] < gain[var_rhs];
                    }
                    return lhs_derived > rhs_derived;
                }        
                return ranges[var_lhs] < ranges[var_rhs];
            });

        if (debug) {
            std::cout << "Choosing var [" << next_var << "] derived? [" << task.get_variables()[next_var].is_derived() << "] with domain size [" << ranges[next_var] <<  "] and gain [" << gain[next_var] << "] (and degree [" << compute_degree(affinity, next_var) << "])" << std::endl;
        }

        bin_vars.push_back(next_var);
        var_infos[next_var] = VariableInfo(ranges[next_var], bin_index, used_bits);
        used_bits += get_bit_size_for_range(ranges[next_var]);
        auto& bit_vars = bits_to_vars[get_bit_size_for_range(ranges[next_var])];
        bit_vars.erase(std::find(bit_vars.begin(), bit_vars.end(), next_var));
        ++num_vars_in_bin;
        unpacked_vars.erase(next_var);
        fit_unpacked_vars.erase(next_var);
    }
}

}
