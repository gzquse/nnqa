#!/usr/bin/env python3
"""
IonQ Stress Test Results Plotter
=================================

Generates scaling analysis plots for stress test results showing performance
vs polynomial degree up to 20.

Usage:
    python plot_stress_test.py --input results/stress_test --output figures/stress_test
"""

import sys
import os
import numpy as np
import glob
import argparse
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# Add parent paths
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from toolbox.Util_H5io4 import read4_data_hdf5
from ionq_config import POLYNOMIALS, PLOT_CONFIG, evaluate_polynomial

# ==============================================================================
# MATPLOTLIB CONFIGURATION
# ==============================================================================

def configure_publication_style():
    """Configure matplotlib for publication-quality figures."""
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'DejaVu Serif', 'Computer Modern Roman'],
        'font.size': 11,
        'axes.labelsize': 12,
        'axes.titlesize': 13,
        'legend.fontsize': 9,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'figure.dpi': 150,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.05,
        'axes.linewidth': 1.0,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'lines.linewidth': 2.0,
        'lines.markersize': 6,
        'legend.framealpha': 0.95,
        'legend.edgecolor': '0.8',
        'legend.fancybox': False,
        'grid.alpha': 0.3,
        'grid.linewidth': 0.5,
    })

# ==============================================================================
# DATA LOADING
# ==============================================================================

def load_stress_test_results(input_dir):
    """Load all H5 result files from stress test."""
    h5_files = glob.glob(os.path.join(input_dir, 'stress_deg*.h5'))
    
    all_data = []
    for h5_file in sorted(h5_files):
        try:
            data, meta = read4_data_hdf5(h5_file, verb=0)
            
            # Compute metrics if not present
            if 'metrics' not in meta:
                theoretical = data['theoretical']
                classical_pred = data['classical_pred']
                measured = data['measured']
                
                quantum_rmse = np.sqrt(np.mean((measured - theoretical) ** 2))
                classical_rmse = np.sqrt(np.mean((classical_pred - theoretical) ** 2))
                quantum_corr = np.corrcoef(theoretical, measured)[0, 1]
                classical_corr = np.corrcoef(theoretical, classical_pred)[0, 1]
                
                abs_errors = np.abs(measured - theoretical)
                pass_rate = np.mean(abs_errors < 0.03)
                poor_rate = np.mean((abs_errors >= 0.03) & (abs_errors < 0.1))
                fail_rate = np.mean(abs_errors >= 0.1)
                
                meta['metrics'] = {
                    'quantum_rmse': float(quantum_rmse),
                    'classical_rmse': float(classical_rmse),
                    'quantum_corr': float(quantum_corr),
                    'classical_corr': float(classical_corr),
                    'pass_rate': float(pass_rate),
                    'poor_rate': float(poor_rate),
                    'fail_rate': float(fail_rate),
                }
            
            result = {
                'degree': meta['degree'],
                'trial': meta['trial'],
                'polynomial_name': meta.get('polynomial_name', f'Degree {meta["degree"]}'),
                'metrics': meta['metrics'],
                'backend': meta.get('backend', 'ionq_forte-1'),
                'provider': meta.get('provider', 'ionq'),
                'x_values': data['x_values'],
                'theoretical': data['theoretical'],
                'classical_pred': data['classical_pred'],
                'measured': data['measured'],
                'measured_err': data.get('measured_err', np.zeros_like(data['measured'])),
                'circuit_info': meta.get('circuit_info', []),
            }
            all_data.append(result)
        except Exception as e:
            print(f"Warning: Skipping {h5_file} due to error: {e}")
            continue
    
    print(f"Loaded {len(all_data)} stress test result files from {input_dir}")
    return all_data

def aggregate_by_degree(all_data):
    """Aggregate results by polynomial degree."""
    degrees = sorted(set(r['degree'] for r in all_data))
    
    aggregated = {}
    for degree in degrees:
        degree_results = [r for r in all_data if r['degree'] == degree]
        
        q_rmse = [r['metrics']['quantum_rmse'] for r in degree_results]
        c_rmse = [r['metrics']['classical_rmse'] for r in degree_results]
        q_corr = [r['metrics']['quantum_corr'] for r in degree_results]
        c_corr = [r['metrics']['classical_corr'] for r in degree_results]
        pass_rates = [r['metrics']['pass_rate'] for r in degree_results]
        
        # Circuit resources
        circuit_resources = []
        for r in degree_results:
            if r.get('circuit_info'):
                for ci in r['circuit_info']:
                    circuit_resources.append(ci)
        
        aggregated[degree] = {
            'name': POLYNOMIALS[degree]['name'],
            'results': degree_results,
            'n_trials': len(degree_results),
            'quantum_rmse_mean': np.mean(q_rmse),
            'quantum_rmse_std': np.std(q_rmse) if len(q_rmse) > 1 else 0.0,
            'classical_rmse_mean': np.mean(c_rmse),
            'classical_rmse_std': np.std(c_rmse) if len(c_rmse) > 1 else 0.0,
            'quantum_corr_mean': np.mean(q_corr),
            'quantum_corr_std': np.std(q_corr) if len(q_corr) > 1 else 0.0,
            'classical_corr_mean': np.mean(c_corr),
            'classical_corr_std': np.std(c_corr) if len(c_corr) > 1 else 0.0,
            'pass_rate_mean': np.mean(pass_rates),
            'pass_rate_std': np.std(pass_rates) if len(pass_rates) > 1 else 0.0,
            'circuit_resources': circuit_resources,
        }
    
    return aggregated

# ==============================================================================
# FIGURE 1: SCALING ANALYSIS
# ==============================================================================

def create_figure1_scaling_analysis(aggregated, output_path):
    """Create Figure 1: Performance scaling vs degree (up to 20)."""
    configure_publication_style()
    
    colors = PLOT_CONFIG['colors']
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    ax1, ax2, ax3, ax4 = axes.flatten()
    
    degrees = sorted(aggregated.keys())
    x_pos = np.array(degrees)
    
    # Extract data
    q_rmse_mean = [aggregated[d]['quantum_rmse_mean'] for d in degrees]
    q_rmse_std = [aggregated[d]['quantum_rmse_std'] for d in degrees]
    q_corr_mean = [aggregated[d]['quantum_corr_mean'] for d in degrees]
    q_corr_std = [aggregated[d]['quantum_corr_std'] for d in degrees]
    pass_rates = [aggregated[d]['pass_rate_mean'] * 100 for d in degrees]
    pass_std = [aggregated[d]['pass_rate_std'] * 100 for d in degrees]
    
    # Panel A: RMSE vs Degree
    ax1.errorbar(x_pos, q_rmse_mean, yerr=q_rmse_std,
                fmt='o-', color=colors['ionq'], markersize=8,
                capsize=4, capthick=1.5, linewidth=2,
                markeredgecolor='black', markeredgewidth=0.5,
                label='IonQ Quantum')
    ax1.axhline(0.03, color='green', linestyle='--', alpha=0.7, label='PASS threshold')
    ax1.set_xlabel('Polynomial Degree')
    ax1.set_ylabel('RMSE')
    ax1.set_title('(a) Recovery Error vs Degree\n(Stress Test: Up to Degree 20)')
    ax1.set_xticks(degrees)
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Panel B: Correlation vs Degree
    ax2.errorbar(x_pos, q_corr_mean, yerr=q_corr_std,
                fmt='s-', color=colors['ionq'], markersize=8,
                capsize=4, capthick=1.5, linewidth=2,
                markeredgecolor='black', markeredgewidth=0.5,
                label='IonQ Quantum')
    ax2.axhline(0.99, color='orange', linestyle='--', alpha=0.7, label='99% threshold')
    ax2.set_xlabel('Polynomial Degree')
    ax2.set_ylabel('Correlation Coefficient')
    ax2.set_title('(b) Correlation vs Degree\n(Stress Test: Up to Degree 20)')
    ax2.set_xticks(degrees)
    ax2.set_ylim(0.9, 1.01)
    ax2.legend(loc='lower left')
    ax2.grid(True, alpha=0.3)
    
    # Panel C: Pass Rate vs Degree
    ax3.errorbar(x_pos, pass_rates, yerr=pass_std,
                fmt='^-', color=colors['ionq'], markersize=8,
                capsize=4, capthick=1.5, linewidth=2,
                markeredgecolor='black', markeredgewidth=0.5,
                label='Pass Rate')
    ax3.axhline(90, color='orange', linestyle='--', alpha=0.7, label='90% threshold')
    ax3.set_xlabel('Polynomial Degree')
    ax3.set_ylabel('Pass Rate (%)')
    ax3.set_title('(c) Pass Rate vs Degree\n(Stress Test: Up to Degree 20)')
    ax3.set_xticks(degrees)
    ax3.set_ylim(0, 110)
    ax3.legend(loc='lower left')
    ax3.grid(True, alpha=0.3, axis='y')
    
    # Panel D: Circuit Resources vs Degree
    avg_qubits = []
    avg_depth = []
    avg_2q = []
    
    for d in degrees:
        resources = aggregated[d]['circuit_resources']
        if resources:
            avg_qubits.append(np.mean([r['qubits'] for r in resources]))
            avg_depth.append(np.mean([r['depth'] for r in resources]))
            avg_2q.append(np.mean([r['2q_gates'] for r in resources]))
        else:
            # Fallback: estimate from degree
            avg_qubits.append(d)
            avg_depth.append(3 * d)  # Approximate
            avg_2q.append(d - 1)
    
    ax4_twin = ax4.twinx()
    
    line1 = ax4.plot(x_pos, avg_qubits, 'o-', color=colors['theoretical'],
                    linewidth=2, markersize=8, label='Qubits')
    line2 = ax4_twin.plot(x_pos, avg_2q, 's-', color=colors['quantum'],
                          linewidth=2, markersize=8, label='2-Qubit Gates')
    
    ax4.set_xlabel('Polynomial Degree')
    ax4.set_ylabel('Number of Qubits', color=colors['theoretical'])
    ax4_twin.set_ylabel('Number of 2-Qubit Gates', color=colors['quantum'])
    ax4.set_title('(d) Circuit Resources vs Degree\n(Stress Test: Up to Degree 20)')
    ax4.set_xticks(degrees)
    ax4.tick_params(axis='y', labelcolor=colors['theoretical'])
    ax4_twin.tick_params(axis='y', labelcolor=colors['quantum'])
    
    # Combined legend
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax4.legend(lines, labels, loc='upper left')
    
    ax4.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    pdf_path = output_path.rsplit('.', 1)[0] + '.pdf'
    plt.savefig(pdf_path, dpi=300, bbox_inches='tight')
    
    print(f"Figure 1 saved to: {output_path}")
    plt.close()
    return fig

# ==============================================================================
# FIGURE 2: RECOVERY GRID (STRESS TEST DEGREES)
# ==============================================================================

def create_figure2_recovery_grid(aggregated, output_path):
    """Create Figure 2: Recovery grid for stress test degrees."""
    configure_publication_style()
    
    colors = PLOT_CONFIG['colors']
    degrees = sorted(aggregated.keys())
    
    n_degrees = len(degrees)
    n_cols = min(3, n_degrees)
    n_rows = (n_degrees + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 3.5*n_rows))
    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes.reshape(1, -1)
    elif n_cols == 1:
        axes = axes.reshape(-1, 1)
    axes = axes.flatten()
    
    for idx, degree in enumerate(degrees):
        ax = axes[idx]
        data = aggregated[degree]
        
        # Use first trial for visualization
        trial_data = data['results'][0]
        
        x = trial_data['x_values']
        theo = trial_data['theoretical']
        classical = trial_data['classical_pred']
        quantum = trial_data['measured']
        q_err = trial_data['measured_err']
        
        # Dense x for smooth curve
        x_dense = np.linspace(-1, 1, 200)
        coeffs = POLYNOMIALS[degree]['coefficients']
        y_dense = np.array([sum(coeffs[i] * (xv ** i) for i in range(len(coeffs))) 
                          for xv in x_dense])
        
        # Plot theoretical curve
        ax.plot(x_dense, y_dense, color=colors['theoretical'], linewidth=2,
               label='Theoretical', zorder=1)
        
        # Plot classical points
        ax.scatter(x, classical, color=colors['classical'], s=40, marker='s',
                  edgecolors='black', linewidth=0.3, label='Classical', zorder=2, alpha=0.8)
        
        # Plot quantum points with error bars
        ax.errorbar(x, quantum, yerr=q_err, fmt='o', color=colors['ionq'],
                   markersize=6, capsize=3, capthick=1, elinewidth=1,
                   markeredgecolor='black', markeredgewidth=0.3,
                   label='IonQ Quantum', zorder=3, alpha=0.9)
        
        ax.set_xlabel('$x$')
        ax.set_ylabel('$F(x)$')
        ax.set_title(f'Degree {degree}: {data["name"]}\n(Stress Test)')
        ax.set_xlim(-1.05, 1.05)
        ax.grid(True, alpha=0.3)
        
        if idx == 0:
            ax.legend(loc='lower left', fontsize=8)
        
        # Add RMSE annotation
        rmse_text = f'RMSE: {data["quantum_rmse_mean"]:.4f}'
        ax.text(0.95, 0.05, rmse_text, transform=ax.transAxes,
               fontsize=9, ha='right', va='bottom',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Hide extra axes
    for idx in range(len(degrees), len(axes)):
        axes[idx].set_visible(False)
    
    plt.tight_layout()
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    pdf_path = output_path.rsplit('.', 1)[0] + '.pdf'
    plt.savefig(pdf_path, dpi=300, bbox_inches='tight')
    
    print(f"Figure 2 saved to: {output_path}")
    plt.close()
    return fig

# ==============================================================================
# SUMMARY TABLE
# ==============================================================================

def print_summary_table(aggregated):
    """Print a summary table to console."""
    degrees = sorted(aggregated.keys())
    
    print("\n" + "=" * 90)
    print("IONQ STRESS TEST RESULTS SUMMARY (Degrees 1-20)")
    print("=" * 90)
    print(f"{'Deg':<5} {'Name':<15} {'Q-RMSE':<15} {'Corr':<10} {'Pass%':<10} {'Qubits':<8} {'2Q Gates':<10}")
    print("-" * 90)
    
    for degree in degrees:
        data = aggregated[degree]
        q_rmse = f"{data['quantum_rmse_mean']:.4f}"
        corr = f"{data['quantum_corr_mean']:.4f}"
        pass_rate = f"{data['pass_rate_mean']*100:.1f}%"
        
        # Circuit resources
        if data['circuit_resources']:
            avg_qubits = np.mean([r['qubits'] for r in data['circuit_resources']])
            avg_2q = np.mean([r['2q_gates'] for r in data['circuit_resources']])
        else:
            avg_qubits = degree
            avg_2q = degree - 1
        
        print(f"{degree:<5} {data['name']:<15} {q_rmse:<15} {corr:<10} {pass_rate:<10} {avg_qubits:<8.1f} {avg_2q:<10.1f}")
    
    print("=" * 90)

# ==============================================================================
# MAIN
# ==============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description='Generate stress test scaling analysis plots'
    )
    parser.add_argument('--input', default='results/stress_test',
                       help='Input directory with H5 result files')
    parser.add_argument('--output', default='figures/stress_test',
                       help='Output directory for figures')
    parser.add_argument('--all', action='store_true',
                       help='Generate all figures')
    parser.add_argument('--fig1', action='store_true',
                       help='Generate Figure 1: Scaling analysis')
    parser.add_argument('--fig2', action='store_true',
                       help='Generate Figure 2: Recovery grid')
    parser.add_argument('--no-display', action='store_true',
                       help='Do not display plots')
    
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    
    # Resolve paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_dir = os.path.join(script_dir, args.input)
    output_dir = os.path.join(script_dir, args.output)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Load data
    all_data = load_stress_test_results(input_dir)
    if not all_data:
        print("No stress test result data found!")
        print(f"Expected H5 files in: {input_dir}")
        print("\nRun stress test first:")
        print("  python stress_test_ionq.py --execute --submit-only")
        print("  python retrieve_stress_test.py")
        sys.exit(1)
    
    aggregated = aggregate_by_degree(all_data)
    
    # Print summary
    print_summary_table(aggregated)
    
    # Determine which figures to generate
    generate_all = args.all or not (args.fig1 or args.fig2)
    
    print("\n" + "=" * 60)
    print("GENERATING STRESS TEST FIGURES")
    print("=" * 60)
    
    if generate_all or args.fig1:
        fig1_path = os.path.join(output_dir, 'fig1_scaling_analysis.png')
        create_figure1_scaling_analysis(aggregated, fig1_path)
    
    if generate_all or args.fig2:
        fig2_path = os.path.join(output_dir, 'fig2_recovery_grid.png')
        create_figure2_recovery_grid(aggregated, fig2_path)
    
    print("\n" + "=" * 60)
    print(f"All outputs saved to: {output_dir}")
    print("=" * 60)
    
    if not args.no_display:
        plt.show()
