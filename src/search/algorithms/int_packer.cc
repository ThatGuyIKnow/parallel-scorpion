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

int IntPacker::get_bin(int var_id) const {
    return var_infos[var_id].get_bin();
}

int IntPacker::get_range(int var_id) const {
    return var_infos[var_id].get_range();
}

TaskProxy IntPacker::get_task_proxy() const {
    assert(task);
    return TaskProxy(*task);
}


static auto compute_affinity(const TaskProxy &task_proxy, int num_vars) {
    auto affinity = Affinity(num_vars, vector<int>(num_vars, 0));

    for (const auto op : task_proxy.get_operators()) {
        for (const auto eff_lhs : op.get_effects()) {
            for (const auto eff_rhs : op.get_effects()) {
                if (eff_lhs.get_fact().get_variable().get_id() == eff_rhs.get_fact().get_variable().get_id()) continue;

                affinity[eff_lhs.get_fact().get_variable().get_id()]
                        [eff_rhs.get_fact().get_variable().get_id()] += 1;
            }
        }
    }

    return affinity;
}

    static auto compute_degree(const Affinity& affinity, int var)
    {
        int deg = 0;
        for (int val : affinity[var]) {
            deg += val;
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
    auto unpacked_vars = unordered_set<int>();
    for (const auto& var : task_proxy.get_variables()) {
        if (var.is_derived())
            continue;
        int bits = get_bit_size_for_range(ranges[var.get_id()]);
        assert(bits <= BITS_PER_BIN);
        bits_to_vars[bits].push_back(var.get_id());
        unpacked_vars.insert(var.get_id());
    }


    // Step 1: Compute pairwise affinities
    auto affinity = compute_affinity(task_proxy, num_vars);

    auto bin_vars = vector<int>();
    while (!unpacked_vars.empty()) {
        bool is_even = num_bins % 2 == 0;
        if (is_even) {
            pack_one_bin(affinity, unpacked_vars, ranges, bits_to_vars, bin_vars);
        } else {
            pack_one_bin(affinity, unpacked_vars, ranges, bits_to_vars, bin_vars);
            bin_vars.clear();
        }
    }

    // Since derived variables are not relevant to consider in the affinity, we have seperate logic for it
    pack_derived_maxwidth(ranges, task_proxy);
}


int IntPacker::pack_one_bin(const Affinity& affinity,
                            std::unordered_set<int>& unpacked_vars,
                            const vector<int> &ranges,
                            vector<vector<int>> &bits_to_vars,
                            std::vector<int>& bin_vars)
    {
    if (debug)
        std::cout << "Packing bin [" << num_bins << "]" << std::endl;

    ++num_bins;
    int bin_index = num_bins - 1;
    int used_bits = 0;
    int num_vars_in_bin = 0;


    // If there are no variable for consideration, seed a new variable
    if (bin_vars.empty()) {
        const int seed = *std::max_element(unpacked_vars.begin(), unpacked_vars.end(),
                                     [&](int var_lhs, int var_rhs) {
                                         return compute_degree(affinity, var_lhs) < compute_degree(affinity, var_rhs);
                                     });
        var_infos[seed] = VariableInfo(ranges[seed], bin_index, used_bits);
        used_bits += get_bit_size_for_range(ranges[seed]);
        auto& bit_vars = bits_to_vars[get_bit_size_for_range(ranges[seed])];
        bit_vars.erase(std::find(bit_vars.begin(), bit_vars.end(), seed));
        ++num_vars_in_bin;
        bin_vars.push_back(seed);
        unpacked_vars.erase(seed);

        if (debug) {
            std::cout << "Choosing seed [" << seed << "] with degree [" << compute_degree(affinity, seed) << "]" << std::endl;
        }
    }

    while (true) 
    {
        // Determine size of largest variable that still fits into the bin.
        int bits = BITS_PER_BIN - used_bits;
        while (bits > 0 && bits_to_vars[bits].empty())
            --bits;

        if (bits == 0) 
        {
            // No more variables fit into the bin.
            // (This also happens when all variables have been packed.)
            return num_vars_in_bin;
        }

        // Determine variables that fit
        auto fit_unpacked_vars = std::vector<int>{};
        for (int var : unpacked_vars) {
            if (get_bit_size_for_range(ranges[var]) <= bits) {
                fit_unpacked_vars.push_back(var);
            }
        }

        // Compute gain of adding variable into current bin
        auto gain = std::vector<int>(affinity.size(), 0);
        for (int var : fit_unpacked_vars) 
        {
            for (int bin_var : bin_vars) 
            {
                gain[var] += affinity[var][bin_var];
            }
        }

        // Chose variable that maximizes gain, break ties in favor of degree
        const auto next_var = *std::max_element(fit_unpacked_vars.begin(), fit_unpacked_vars.end(),
            [&](int var_lhs, int var_rhs) {
                assert(utils::in_bounds(var_lhs, gain) && utils::in_bounds(var_rhs, gain));
                if (gain[var_lhs] == gain[var_rhs]) 
                {
                    return compute_degree(affinity, var_lhs) < compute_degree(affinity, var_rhs);
                }
                return gain[var_lhs] < gain[var_rhs];
            });

        if (debug) {
            std::cout << "Choosing var [" << next_var << "] with gain [" << gain[next_var] << "] (and degree [" << compute_degree(affinity, next_var) << "])" << std::endl;
        }

        bin_vars.push_back(next_var);
        var_infos[next_var] = VariableInfo(ranges[next_var], bin_index, used_bits);
        used_bits += get_bit_size_for_range(ranges[next_var]);
        auto& bit_vars = bits_to_vars[get_bit_size_for_range(ranges[next_var])];
        bit_vars.erase(std::find(bit_vars.begin(), bit_vars.end(), next_var));
        ++num_vars_in_bin;
        unpacked_vars.erase(next_var);
    }
}

vector<int> IntPacker::get_bins_used_bits() {
    vector<int> bins_used_bits(num_bins, 0);

    for (size_t var = 0; var < var_infos.size(); ++var) {
        const auto& info = var_infos[var];

        if (info.get_range() == 0) {
            continue;
        }

        int bit_size = get_bit_size_for_range(info.get_range());
        bins_used_bits[info.get_bin_index()] += bit_size;
    }

    return bins_used_bits;
}

unordered_map<int, vector<int>> IntPacker::collect_derived_variables_by_bit_size(
    const vector<int>& ranges,
    const TaskProxy& task_proxy) {

    unordered_map<int, vector<int>> derived_bit_to_vars;

    for (const auto& var : task_proxy.get_variables()) {
        if (var.is_derived()) {
            int var_id = var.get_id();
            int bit_size = get_bit_size_for_range(ranges[var_id]);
            derived_bit_to_vars[bit_size].push_back(var_id);
        }
    }

    return derived_bit_to_vars;
}

int IntPacker::fill_bin_with_derived_variables(
    int bin_index,
    int initial_used_bits,
    const vector<int>& ranges,
    unordered_map<int, vector<int>>& bit_to_vars) {

    int used_bits = initial_used_bits;
    int variables_packed = 0;

    while (true) {
        int available_bits = BITS_PER_BIN - used_bits;

        while (available_bits > 0 && bit_to_vars[available_bits].empty()) {
            --available_bits;
        }

        if (available_bits == 0) {
            break;
        }

        int var_id = bit_to_vars[available_bits].back();
        bit_to_vars[available_bits].pop_back();

        var_infos[var_id] = VariableInfo(ranges[var_id], bin_index, used_bits);

        if (debug) {
            cout << "Packing derived variable [" << var_id
                 << "] into bin [" << bin_index << "]" << " with shift [" << used_bits << "]" << endl;
        }

        used_bits += available_bits;
        ++variables_packed;
    }

    return variables_packed;
}

int IntPacker::pack_derived_into_existing_bins(
    const vector<int>& ranges,
    unordered_map<int, vector<int>>& bit_to_vars) {

    vector<int> bins_used_bits = get_bins_used_bits();

    int total_packed = 0;

    for (size_t bin_index = 0; bin_index < bins_used_bits.size(); ++bin_index) {
        int packed = fill_bin_with_derived_variables(
            bin_index,
            bins_used_bits[bin_index],
            ranges,
            bit_to_vars);
        total_packed += packed;
    }

    return total_packed;
}

int IntPacker::pack_derived_into_new_bin(
    const vector<int>& ranges,
    unordered_map<int, vector<int>>& bit_to_vars) {

    int bin_index = num_bins;
    int variables_packed = fill_bin_with_derived_variables(
        bin_index,
        0,
        ranges,
        bit_to_vars);

    ++num_bins;
    return variables_packed;
}

void IntPacker::pack_derived_maxwidth(
    const vector<int>& ranges,
    const TaskProxy& task_proxy) {

    unordered_map<int, vector<int>> bit_to_vars =
        collect_derived_variables_by_bit_size(ranges, task_proxy);


    int remaining_derived = 0;
    for (const auto& [bit_size, vars] : bit_to_vars) {
        remaining_derived += vars.size();
    }

    remaining_derived -= pack_derived_into_existing_bins(ranges, bit_to_vars);

    while (remaining_derived > 0)
        remaining_derived -= pack_derived_into_new_bin(ranges, bit_to_vars);
}
}
