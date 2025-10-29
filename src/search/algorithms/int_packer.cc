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
#include <clingo.hh>
#include <sstream>

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

TaskProxy IntPacker::get_task_proxy() const {
    assert(task);
    return TaskProxy(*task);
}


// Structure to represent a bin during packing
struct BinInfo {
    unordered_set<int> vars;
    int used_bits;

    BinInfo() : used_bits(0) {}
};

int IntPacker::get_min_bins(const vector<int>& ranges) const {
    // bits_to_vars[k] contains all variables that require exactly k
    // bits to encode. Once a variable is packed into a bin, it is
    // removed from this index.
    // Loop over the variables in reverse order to prefer variables with
    // low indices in case of ties. This might increase cache-locality.
    int num_vars = ranges.size();
    vector<int> bits_to_vars(BITS_PER_BIN + 1);
    for (int var = num_vars - 1; var >= 0; --var) {
        int bits = get_bit_size_for_range(ranges[var]);
        assert(bits <= BITS_PER_BIN);
        bits_to_vars[bits]++;
    }

    int num_bins = 1;  // Start with 1 to count the first bin
    int used_bits = 0;
    int packed_vars = 0;
    while (true) {
        if (packed_vars == num_vars)
            return num_bins;

        int bits_left = BITS_PER_BIN - used_bits;
        while (bits_left > 0 && bits_to_vars[bits_left] == 0)
            --bits_left;

        if (bits_left == 0) {
            num_bins++;
            used_bits = 0;
            continue;  // Start over to find next variable for the new bin
        }

        bits_to_vars[bits_left]--;
        used_bits += bits_left;
        packed_vars++;
    }
}

static std::string generate_fact_sheet(const vector<int> &ranges,
        const TaskProxy &task_proxy,
        const int num_bins,
        const int bin_size) {
    stringstream fact_sheet("");

    fact_sheet << "#const max_bins = " << num_bins << ".\n";
    fact_sheet << "#const bin_bit_width = " << bin_size << ".\n";

    for (const auto& var : task_proxy.get_variables()) {
        const auto var_id = var.get_id();
        fact_sheet << "variable(v" << var_id << ").\n";
        fact_sheet << "bit_width(v" << var_id << ", " << get_bit_size_for_range(ranges[var_id]) << ").\n";
    }

    for (const auto& op : task_proxy.get_operators()) {
        fact_sheet << "operator(o" << op.get_id() << ").\n";
        for (const auto& eff : op.get_effects()) {
            fact_sheet << "effect(o" << op.get_id() << ", v" << eff.get_fact().get_variable().get_id() << ").\n";
        }
    }

    return fact_sheet.str();
}

vector<pair<int, int>> IntPacker::find_min_operator_variable_packing(const vector<int> &ranges) {

    const auto task_proxy = get_task_proxy();
    const auto fact_sheet = generate_fact_sheet(ranges, task_proxy, num_bins, sizeof(Bin) * 8);

    Clingo::Logger logger = [](Clingo::WarningCode, char const *msg) {
        std::cerr << "Warning: " << msg << std::endl;
    };

    Clingo::Control ctl{{"--opt-mode=optN"}, logger};
    // Add the fact sheet string to the "base" program part
    // Parameters: program_name, parameters, program_string
    ctl.add("base", {}, fact_sheet.c_str());

    // Load problem definition from file (or also as string)
    ctl.load("src/search/algorithms/opt_operator_variable_bin.lp");

    // Ground the program
    ctl.ground({{"base", {}}});
    // Storage for results
    vector<pair<int, int>> throw_pairs;

    std::cout << " === Solving === " << std::endl;
    for (auto &model : ctl.solve()) {
        throw_pairs.clear();
        std::cout << "Iterating..." << std::endl;

        for (auto &atom : model.symbols(Clingo::ShowType::Shown)) {
            if (atom.type() == Clingo::SymbolType::Function &&
                std::string(atom.name()) == "throw") {
                auto args = atom.arguments();
                // v[\d] < captures the variable index
                // Convert C-style names to std::string before using substr/stoi.
                const std::string arg0_name(args[0].name());
                const auto var_idx = std::stoi(arg0_name.substr(1));
                const auto bin_idx = args[1].number();
                throw_pairs.emplace_back(var_idx, bin_idx);
            }
        }

        std::cout << " - Optimal: " << (model.optimality_proven() ? "YES" : "NO") << std::endl;
        if (model.optimality_proven())
            break;
    }

    std::cout << "Throw pairs: [ ";
    for (const auto& pair : throw_pairs) {
        std::cout << "(" << pair.first << ", " << pair.second << "), ";
    }
    std::cout << "]" << std::endl;
    return throw_pairs;
}

void IntPacker::pack_bins(const vector<int> &ranges) {
    assert(var_infos.empty());

    std::cout << "Using optimized bin packing algorithm" << std::endl;
    int num_vars = ranges.size();
    var_infos.resize(num_vars);
    num_bins = get_min_bins(ranges);
    const auto var_assignments = find_min_operator_variable_packing(ranges);

    auto used_bits = vector<int>(num_bins, 0);
    for (const auto& [var_idx, bin_idx] : var_assignments) {
        var_infos[var_idx] = VariableInfo(ranges[var_idx], bin_idx, used_bits[bin_idx]);
        if (debug) {
            std::cout << "Packing variable [" << var_idx
                      << "] (range " << ranges[var_idx] << ") into bin ["
                      << bin_idx << "]" << " at shift [" << used_bits[bin_idx] << "]" << std::endl;
        }
        used_bits[bin_idx] += get_bit_size_for_range(ranges[var_idx]);
    }

    std::cout << "Packed " << num_vars << " variables into " << num_bins << " bins" << std::endl;
}



}
