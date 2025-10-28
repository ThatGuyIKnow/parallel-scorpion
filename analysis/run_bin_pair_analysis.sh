#!/bin/bash

# Script to run domain analysis and parse bin pair packing results
# Runs problems sequentially to avoid output issues
# Usage: ./run_bin_pair_analysis.sh [domain_name]
# Example: ./run_bin_pair_analysis.sh depot
#          ./run_bin_pair_analysis.sh gripper
#          ./run_bin_pair_analysis.sh blocksworld

# Configuration
BENCHMARK_DIR="${DOWNWARD_BENCHMARKS:-$HOME/benchmarks}"
DOMAIN_NAME="${1:-depot}"  # Default to depot if not specified
DOMAIN_DIR="$BENCHMARK_DIR/$DOMAIN_NAME"
OUTPUT_DIR="./${DOMAIN_NAME}_bin_pair_analysis_results"
RESULTS_FILE="$OUTPUT_DIR/bin_pair_packing_results.csv"
SEARCH_TIME_LIMIT="1s"

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Initialize results CSV file
echo "Problem,Strategy,NumBins,NumBinPairs,AvgBinPairsTouched,MaxBinPairsTouched" > "$RESULTS_FILE"

# Check if domain directory exists
if [ ! -d "$DOMAIN_DIR" ]; then
    echo "Error: Domain directory not found at $DOMAIN_DIR"
    echo "Please specify a valid domain name"
    echo "Usage: $0 [domain_name]"
    echo "Example: $0 depot"
    echo ""
    echo "Available domains in $BENCHMARK_DIR:"
    ls -d "$BENCHMARK_DIR"/*/ 2>/dev/null | xargs -n 1 basename | head -20
    exit 1
fi

# Find all problems in the domain
PROBLEMS=($(find "$DOMAIN_DIR" -name "*.pddl" -type f | grep -E "p[0-9]+\.pddl$" | sort -V))

if [ ${#PROBLEMS[@]} -eq 0 ]; then
    echo "Error: No problems found in $DOMAIN_DIR"
    exit 1
fi

echo "Running bin pair analysis for domain: $DOMAIN_NAME"
echo "Found ${#PROBLEMS[@]} problems"
echo "Output directory: $OUTPUT_DIR"
echo "Results file: $RESULTS_FILE"
echo ""

# Run each problem sequentially
for problem_file in "${PROBLEMS[@]}"; do
    problem_name=$(basename "$problem_file" .pddl)
    log_file="$OUTPUT_DIR/${problem_name}.log"

    echo "Running $problem_name..."

    # Run Fast Downward with 1s time limit
    ../fast-downward.py --search-time-limit "$SEARCH_TIME_LIMIT" \
        "$problem_file" \
        --search "astar(blind())" \
        > "$log_file" 2>&1 || true

    # Parse the bin packing comparison section
    if grep -q "Bin Packing Comparison" "$log_file"; then
        echo "  Parsing results for $problem_name"

        # Use a simpler, more robust parsing approach
        grep -A 20 "Bin Packing Comparison" "$log_file" | \
        awk -v problem="$problem_name" '
        /\[1\] Original Greedy/ { strategy="Greedy"; getline; next }
        /\[2\] Affinity-based/ { strategy="Affinity"; getline; next }

        /Number of bins:/ {
            bins = $NF
        }
        /Number of bin pairs:/ {
            bin_pairs = $NF
        }
        /Average bin pairs touched per operator:/ {
            avg = $NF
        }
        /Maximum bin pairs touched by any operator:/ {
            max = $NF
            if (strategy != "" && bins != "" && bin_pairs != "" && avg != "" && max != "") {
                print problem "," strategy "," bins "," bin_pairs "," avg "," max
                strategy = ""; bins = ""; bin_pairs = ""; avg = ""; max = ""
            }
        }
        ' >> "$RESULTS_FILE"

        echo "  ✓ Completed $problem_name"
    else
        echo "  ⚠ No bin packing comparison found in $problem_name"
    fi
    echo ""
done

echo ""
echo "All problems completed!"
echo "Results saved to: $RESULTS_FILE"
echo ""

# Generate summary statistics
echo "=== Summary Statistics ==="
echo ""

for strategy in "Greedy" "Affinity"; do
    echo "Strategy: $strategy"
    awk -F',' -v strat="$strategy" '
    NR > 1 && $2 == strat {
        count++
        bins_sum += $3
        bin_pairs_sum += $4
        avg_sum += $5
        max_sum += $6
        if (max_max == "" || $6 > max_max) max_max = $6
    }
    END {
        if (count > 0) {
            printf "  Problems: %d\n", count
            printf "  Avg Number of Bins: %.2f\n", bins_sum/count
            printf "  Avg Number of Bin Pairs: %.2f\n", bin_pairs_sum/count
            printf "  Avg Bin Pairs Touched: %.2f\n", avg_sum/count
            printf "  Avg Max Bin Pairs Touched: %.2f\n", max_sum/count
            printf "  Overall Max Bin Pairs Touched: %d\n", max_max
        }
    }
    ' "$RESULTS_FILE"
    echo ""
done

echo "Detailed results available in: $OUTPUT_DIR/"
echo ""
echo "First few results:"
head -n 10 "$RESULTS_FILE"
