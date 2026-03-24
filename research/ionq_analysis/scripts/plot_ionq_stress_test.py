#!/usr/bin/env python3
"""
IonQ Stress Test Plotting Script
=================================

Generates publication-quality figures for IonQ Forte-1 stress test results.
Plots scaling analysis and recovery grids for polynomial degrees 1-35.

Usage:
    python plot_ionq_stress_test.py --all
    python plot_ionq_stress_test.py --fig1 --fig2
"""

import sys
import os

# Add paths to find dependencies
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(os.path.dirname(script_dir)))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'research', 'ionq'))

# Import the plotting functions from the original script
import importlib.util
plot_module_path = os.path.join(project_root, 'research', 'ionq', 'plot_stress_test.py')
spec = importlib.util.spec_from_file_location("plot_stress_test", plot_module_path)
plot_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(plot_module)

load_stress_test_results = plot_module.load_stress_test_results
aggregate_by_degree = plot_module.aggregate_by_degree
create_figure1_scaling_analysis = plot_module.create_figure1_scaling_analysis
create_figure2_recovery_grid = plot_module.create_figure2_recovery_grid
print_summary_table = plot_module.print_summary_table

import argparse

def main():
    """Main plotting function for IonQ stress test."""
    parser = argparse.ArgumentParser(
        description='Generate IonQ stress test figures'
    )
    parser.add_argument('--input', default='../data',
                       help='Input directory with H5 result files')
    parser.add_argument('--output', default='../figures',
                       help='Output directory for figures')
    parser.add_argument('--all', action='store_true',
                       help='Generate all figures')
    parser.add_argument('--fig1', action='store_true',
                       help='Generate Figure 1: Scaling analysis')
    parser.add_argument('--fig2', action='store_true',
                       help='Generate Figure 2: Recovery grid')
    parser.add_argument('--no-display', action='store_true', default=True,
                       help='Do not display plots (default: True)')
    
    args = parser.parse_args()
    
    # Resolve paths relative to script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_dir = os.path.abspath(os.path.join(script_dir, args.input))
    output_dir = os.path.abspath(os.path.join(script_dir, args.output))
    
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 70)
    print("IONQ STRESS TEST PLOTTING")
    print("=" * 70)
    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")
    
    # Load data
    all_data = load_stress_test_results(input_dir)
    if not all_data:
        print(f"\nERROR: No stress test result data found!")
        print(f"Expected H5 files in: {input_dir}")
        print("\nPlease retrieve results first:")
        print("  python retrieve_ionq_stress_test.py")
        sys.exit(1)
    
    aggregated = aggregate_by_degree(all_data)
    
    # Print summary
    print_summary_table(aggregated)
    
    # Determine which figures to generate
    generate_all = args.all or not (args.fig1 or args.fig2)
    
    print("\n" + "=" * 70)
    print("GENERATING FIGURES")
    print("=" * 70)
    
    if generate_all or args.fig1:
        fig1_path = os.path.join(output_dir, 'fig1_scaling_analysis.pdf')
        create_figure1_scaling_analysis(aggregated, fig1_path)
        print(f"✓ Figure 1: Scaling analysis saved")
    
    if generate_all or args.fig2:
        fig2_path = os.path.join(output_dir, 'fig2_recovery_grid.pdf')
        create_figure2_recovery_grid(aggregated, fig2_path)
        print(f"✓ Figure 2: Recovery grid saved")
    
    print("\n" + "=" * 70)
    print(f"All outputs saved to: {output_dir}")
    print("=" * 70)

if __name__ == '__main__':
    main()
