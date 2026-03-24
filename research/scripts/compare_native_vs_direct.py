#!/usr/bin/env python3
"""
Compare Native vs Direct Approach Results
==========================================

Generates comparison plots and results table for:
- Native approach: Multi-qubit quantum arithmetic (maps NN weights to quantum polynomial protocol)
- Direct approach: Single-qubit encoding (classical computation + quantum encoding)

Usage:
    python compare_native_vs_direct.py --native-dir results/cloud --direct-dir results/direct --output figures/paper/comparison
"""

import sys
import os
import numpy as np
import glob
import argparse
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from research_config import POLYNOMIALS, PLOT_CONFIG
from toolbox.Util_H5io4 import read4_data_hdf5


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


def load_results(input_dir, approach_name):
    """Load all H5 result files from directory."""
    h5_files = glob.glob(os.path.join(input_dir, 'deg*.h5'))
    h5_files = [f for f in h5_files if '_submitted' not in f]
    
    all_data = []
    for h5_file in sorted(h5_files):
        try:
            data, meta = read4_data_hdf5(h5_file, verb=0)
            
            # Check approach
            approach = meta.get('approach', 'unknown')
            # If approach not set, infer from directory or assume based on context
            if approach == 'unknown':
                if 'direct' in input_dir.lower():
                    approach = 'direct'
                elif 'cloud' in input_dir.lower() or 'native' in input_dir.lower():
                    approach = 'native'
            
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
                'approach': approach,
                'x_values': data['x_values'],
                'theoretical': data['theoretical'],
                'classical_pred': data['classical_pred'],
                'measured': data['measured'],
                'measured_err': data.get('measured_err', np.zeros_like(data['measured'])),
            }
            all_data.append(result)
        except Exception as e:
            print(f"Warning: Skipping {h5_file} due to error: {e}")
            continue
    
    print(f"Loaded {len(all_data)} {approach_name} result files from {input_dir}")
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
        
        aggregated[degree] = {
            'name': POLYNOMIALS[degree]['name'],
            'results': degree_results,
            'n_trials': len(degree_results),
            'quantum_rmse_mean': np.mean(q_rmse),
            'quantum_rmse_std': np.std(q_rmse),
            'classical_rmse_mean': np.mean(c_rmse),
            'classical_rmse_std': np.std(c_rmse),
            'quantum_corr_mean': np.mean(q_corr),
            'quantum_corr_std': np.std(q_corr),
            'classical_corr_mean': np.mean(c_corr),
            'classical_corr_std': np.std(c_corr),
            'pass_rate_mean': np.mean(pass_rates),
            'pass_rate_std': np.std(pass_rates),
        }
    
    return aggregated


def create_comparison_figure(native_agg, direct_agg, output_path):
    """Create comparison figure showing both approaches."""
    configure_publication_style()
    
    colors = PLOT_CONFIG['colors']
    degrees = sorted(set(list(native_agg.keys()) + list(direct_agg.keys())))
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Panel A: RMSE Comparison
    ax1 = axes[0, 0]
    x_pos = np.array(degrees)
    width = 0.35
    
    native_rmse = [native_agg[d]['quantum_rmse_mean'] for d in degrees]
    native_rmse_std = [native_agg[d]['quantum_rmse_std'] for d in degrees]
    direct_rmse = [direct_agg[d]['quantum_rmse_mean'] for d in degrees]
    direct_rmse_std = [direct_agg[d]['quantum_rmse_std'] for d in degrees]
    
    ax1.bar(x_pos - width/2, direct_rmse, width, yerr=direct_rmse_std,
           color=colors['classical'], label='Direct (1-qubit)',
           edgecolor='black', linewidth=0.5, capsize=3)
    ax1.bar(x_pos + width/2, native_rmse, width, yerr=native_rmse_std,
           color=colors['quantum'], label='Native (multi-qubit)',
           edgecolor='black', linewidth=0.5, capsize=3)
    
    ax1.set_xlabel('Polynomial Degree')
    ax1.set_ylabel('RMSE')
    ax1.set_title('(a) Recovery Error: Direct vs Native')
    ax1.set_xticks(degrees)
    ax1.legend()
    ax1.axhline(0.03, color='green', linestyle='--', alpha=0.5, linewidth=1)
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Panel B: Correlation Comparison
    ax2 = axes[0, 1]
    
    native_corr = [native_agg[d]['quantum_corr_mean'] for d in degrees]
    native_corr_std = [native_agg[d]['quantum_corr_std'] for d in degrees]
    direct_corr = [direct_agg[d]['quantum_corr_mean'] for d in degrees]
    direct_corr_std = [direct_agg[d]['quantum_corr_std'] for d in degrees]
    
    ax2.errorbar(x_pos - 0.1, direct_corr, yerr=direct_corr_std,
                fmt='s', color=colors['classical'], markersize=8,
                capsize=4, capthick=1.5, label='Direct (1-qubit)',
                markeredgecolor='black', markeredgewidth=0.5)
    ax2.errorbar(x_pos + 0.1, native_corr, yerr=native_corr_std,
                fmt='o', color=colors['quantum'], markersize=8,
                capsize=4, capthick=1.5, label='Native (multi-qubit)',
                markeredgecolor='black', markeredgewidth=0.5)
    
    ax2.set_xlabel('Polynomial Degree')
    ax2.set_ylabel('Correlation Coefficient')
    ax2.set_title('(b) Correlation: Direct vs Native')
    ax2.set_xticks(degrees)
    ax2.legend()
    ax2.set_ylim(0.99, 1.001)
    ax2.grid(True, alpha=0.3)
    
    # Panel C: Pass Rate Comparison
    ax3 = axes[1, 0]
    
    native_pass = [native_agg[d]['pass_rate_mean'] * 100 for d in degrees]
    native_pass_std = [native_agg[d]['pass_rate_std'] * 100 for d in degrees]
    direct_pass = [direct_agg[d]['pass_rate_mean'] * 100 for d in degrees]
    direct_pass_std = [direct_agg[d]['pass_rate_std'] * 100 for d in degrees]
    
    ax3.bar(x_pos - width/2, direct_pass, width, yerr=direct_pass_std,
           color=colors['classical'], label='Direct (1-qubit)',
           edgecolor='black', linewidth=0.5, capsize=3)
    ax3.bar(x_pos + width/2, native_pass, width, yerr=native_pass_std,
           color=colors['quantum'], label='Native (multi-qubit)',
           edgecolor='black', linewidth=0.5, capsize=3)
    
    ax3.set_xlabel('Polynomial Degree')
    ax3.set_ylabel('Pass Rate (%)')
    ax3.set_title('(c) Pass Rate: Direct vs Native')
    ax3.set_xticks(degrees)
    ax3.legend()
    ax3.set_ylim(0, 110)
    ax3.grid(True, alpha=0.3, axis='y')
    
    # Panel D: Example Recovery (Degree 3)
    ax4 = axes[1, 1]
    
    if 3 in degrees and native_agg[3]['results']:
        # Use first trial from native approach
        native_trial = native_agg[3]['results'][0]
        
        x = native_trial['x_values']
        theo = native_trial['theoretical']
        
        ax4.plot(x, theo, 'k-', linewidth=2, label='Theoretical', zorder=1)
        
        # Plot direct if available
        if direct_agg[3]['results']:
            direct_trial = direct_agg[3]['results'][0]
            ax4.errorbar(x, direct_trial['measured'], yerr=direct_trial['measured_err'],
                        fmt='s', color=colors['classical'], markersize=6,
                        capsize=3, label='Direct (1-qubit)', zorder=2, alpha=0.8)
        
        ax4.errorbar(x, native_trial['measured'], yerr=native_trial['measured_err'],
                    fmt='o', color=colors['quantum'], markersize=6,
                    capsize=3, label='Native (multi-qubit)', zorder=3, alpha=0.8)
        
        ax4.set_xlabel('$x$')
        ax4.set_ylabel('$F(x)$')
        ax4.set_title('(d) Example Recovery (Degree 3)')
        ax4.legend(fontsize=8)
        ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    pdf_path = output_path.rsplit('.', 1)[0] + '.pdf'
    plt.savefig(pdf_path, dpi=300, bbox_inches='tight')
    
    print(f"Comparison figure saved to: {output_path}")
    plt.close()


def generate_comparison_table(native_agg, direct_agg, output_path):
    """Generate LaTeX table comparing both approaches."""
    degrees = sorted(set(list(native_agg.keys()) + list(direct_agg.keys())))
    
    latex = r"""
\begin{table}[ht]
\centering
\caption{Polynomial Recovery: Direct (1-qubit) vs Native (multi-qubit) Approaches}
\label{tab:comparison}
\begin{tabular}{lcccccc}
\toprule
Degree & Approach & Q-RMSE & Correlation & Pass Rate & Qubits & Gates/Iter \\
\midrule
"""
    
    for degree in degrees:
        native = native_agg[degree]
        direct = direct_agg[degree]
        
        # Native row
        native_rmse = f"${native['quantum_rmse_mean']:.4f} \\pm {native['quantum_rmse_std']:.4f}$"
        native_corr = f"${native['quantum_corr_mean']:.4f}$"
        native_pass = f"${native['pass_rate_mean']*100:.1f}\\%$"
        native_qubits = degree + 1
        native_gates = 2 * degree - 1
        
        latex += f"{degree} & Native & {native_rmse} & {native_corr} & {native_pass} & {native_qubits} & {native_gates} \\\\\n"
        
        # Direct row
        direct_rmse = f"${direct['quantum_rmse_mean']:.4f} \\pm {direct['quantum_rmse_std']:.4f}$"
        direct_corr = f"${direct['quantum_corr_mean']:.4f}$"
        direct_pass = f"${direct['pass_rate_mean']*100:.1f}\\%$"
        direct_qubits = 1
        direct_gates = 1
        
        latex += f" & Direct & {direct_rmse} & {direct_corr} & {direct_pass} & {direct_qubits} & {direct_gates} \\\\\n"
        latex += r"\midrule" + "\n"
    
    latex += r"""\bottomrule
\end{tabular}
\end{table}
"""
    
    with open(output_path, 'w') as f:
        f.write(latex)
    
    print(f"Comparison table saved to: {output_path}")


def parse_args():
    parser = argparse.ArgumentParser(
        description='Compare native vs direct approach results'
    )
    parser.add_argument('--native-dir', default='results/cloud',
                       help='Directory with native approach results')
    parser.add_argument('--direct-dir', default='results/direct',
                       help='Directory with direct approach results')
    parser.add_argument('--output', default='figures/paper/comparison',
                       help='Output directory for figures')
    parser.add_argument('--no-display', action='store_true',
                       help='Do not display plots')
    
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    native_dir = os.path.join(script_dir, args.native_dir)
    direct_dir = os.path.join(script_dir, args.direct_dir)
    output_dir = os.path.join(script_dir, args.output)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Load results
    print("Loading results...")
    native_data = load_results(native_dir, 'native')
    direct_data = load_results(direct_dir, 'direct')
    
    if not native_data:
        print("ERROR: No native approach data found!")
        sys.exit(1)
    
    if not direct_data:
        print("WARNING: No direct approach data found. Using native data as placeholder.")
        print("         For proper comparison, submit direct approach experiments first.")
        direct_data = []  # Will create empty aggregation
    
    # Aggregate
    native_agg = aggregate_by_degree(native_data)
    
    if direct_data:
        direct_agg = aggregate_by_degree(direct_data)
    else:
        # Create empty aggregation with same structure
        direct_agg = {}
        for degree in native_agg.keys():
            direct_agg[degree] = {
                'name': native_agg[degree]['name'],
                'results': [],
                'n_trials': 0,
                'quantum_rmse_mean': 0.0,
                'quantum_rmse_std': 0.0,
                'classical_rmse_mean': 0.0,
                'classical_rmse_std': 0.0,
                'quantum_corr_mean': 0.0,
                'quantum_corr_std': 0.0,
                'classical_corr_mean': 0.0,
                'classical_corr_std': 0.0,
                'pass_rate_mean': 0.0,
                'pass_rate_std': 0.0,
            }
    
    # Generate comparison figure
    fig_path = os.path.join(output_dir, 'comparison_native_vs_direct.png')
    create_comparison_figure(native_agg, direct_agg, fig_path)
    
    # Generate comparison table
    table_path = os.path.join(output_dir, 'comparison_table.tex')
    generate_comparison_table(native_agg, direct_agg, table_path)
    
    print(f"\nAll outputs saved to: {output_dir}")
