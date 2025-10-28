#!/usr/bin/env python3
import pandas as pd
import matplotlib.pyplot as plt
import sys

# Read command line argument for domain name, default to depot
domain = sys.argv[1] if len(sys.argv) > 1 else 'depot'
csv_file = f'./{domain}_analysis_results/bin_packing_results.csv'

try:
    # Read the CSV file
    df = pd.read_csv(csv_file)
    
    # Pivot data to get Greedy and Affinity side by side
    greedy = df[df['Strategy'] == 'Greedy'].set_index('Problem')
    affinity = df[df['Strategy'] == 'Affinity'].set_index('Problem')
    
    # Merge on problem names
    merged = greedy.join(affinity, lsuffix='_Greedy', rsuffix='_Affinity')
    
    # Create scatter plot
    plt.figure(figsize=(10, 8))
    plt.scatter(merged['AvgBinsTouched_Greedy'],
                merged['AvgBinsTouched_Affinity'],
                alpha=0.6, s=100)
    
    # Add diagonal line (y=x) for reference
    max_val = max(merged['AvgBinsTouched_Greedy'].max(),
                  merged['AvgBinsTouched_Affinity'].max())
    plt.plot([0, max_val], [0, max_val], 'r--', alpha=0.5, label='Equal performance')
    
    # Labels and title
    plt.xlabel('Greedy Packing - Avg Num Bins Affected by Operator', fontsize=12)
    plt.ylabel('Affinity Packing - Avg Num Bins Affected by Operator', fontsize=12)
    plt.title(f'Bin Packing Performance Comparison - {domain.capitalize()} Domain', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Add text showing how many points are below the line (Affinity wins)
    below_line = (merged['AvgBinsTouched_Affinity'] < merged['AvgBinsTouched_Greedy']).sum()
    total = len(merged)
    plt.text(0.02, 0.98, f'Affinity wins: {below_line}/{total} problems ({100*below_line/total:.1f}%)',
             transform=plt.gca().transAxes, fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Save and show
    output_file = f'./{domain}_analysis_results/comparison_plot.png'
    plt.tight_layout()
    plt.savefig(output_file, dpi=150)
    print(f"\nPlot saved to: {output_file}")
    plt.show()
    
    # Print summary statistics
    print(f"\n=== Summary for {domain} domain ===")
    print(f"Number of problems: {total}")
    print(f"Affinity wins: {below_line} ({100*below_line/total:.1f}%)")
    print(f"Greedy avg: {merged['AvgBinsTouched_Greedy'].mean():.3f}")
    print(f"Affinity avg: {merged['AvgBinsTouched_Affinity'].mean():.3f}")
    print(f"Improvement: {100*(1 - merged['AvgBinsTouched_Affinity'].mean()/merged['AvgBinsTouched_Greedy'].mean()):.1f}%")

except FileNotFoundError:
    print(f"Error: Could not find {csv_file}")
    print(f"Please run ./run_depot_analysis.sh {domain} first")
    sys.exit(1)
except Exception as e:
    print(f"Error: {e}")
    sys.exit(1)
