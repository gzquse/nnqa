#!/usr/bin/env python3
"""
Native Polynomial Protocol: Publication-Quality Figures
=======================================================

Generates 4 main figures and LaTeX tables specifically for the native polynomial
protocol (multi-qubit quantum arithmetic approach) from cloud results.

This script focuses on results from the native approach where polynomial powers
are computed using quantum multiplication primitives.

Usage:
    python plot_native_polynomial.py --input results/cloud --output figures/paper/native
    python plot_native_polynomial.py --all
"""

import sys
import os
import numpy as np
import json
import glob
import argparse
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from research_config import POLYNOMIALS, PLOT_CONFIG, format_polynomial_latex
from toolbox.Util_H5io4 import read4_data_hdf5


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

def load_native_results(input_dir):
    """
    Load all H5 result files from directory, filtering for native approach.
    
    Cloud results from submit_cloud_batch.py use native approach (multi-qubit).
    If metadata doesn't have 'approach' field, assume native for cloud results.
    """
    h5_files = glob.glob(os.path.join(input_dir, 'deg*.h5'))
    # Exclude submitted files
    h5_files = [f for f in h5_files if '_submitted' not in f]
    
    all_data = []
    for h5_file in sorted(h5_files):
        try:
            data, meta = read4_data_hdf5(h5_file, verb=0)
            
            # Check approach - if not specified, assume native for cloud results
            approach = meta.get('approach', 'native')
            # If approach is missing, check if it's from cloud (has backend)
            if 'approach' not in meta and 'backend' in meta:
                approach = 'native'  # Cloud results use native approach
            
            # Filter: only include native approach results
            if approach != 'native':
                continue
            
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
            
            # Combine data and meta
            result = {
                'degree': meta['degree'],
                'trial': meta['trial'],
                'polynomial_name': meta.get('polynomial_name', f'Degree {meta["degree"]}'),
                'metrics': meta['metrics'],
                'config': meta.get('config', {}),
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
    
    print(f"Loaded {len(all_data)} native approach result files from {input_dir}")
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


# ==============================================================================
# FIGURE 1: DEGREE SCALING
# ==============================================================================

def create_figure1_degree_scaling(aggregated, output_path):
    """Create Figure 1: RMSE vs polynomial degree with error bars (Native Protocol)."""
    configure_publication_style()
    
    colors = PLOT_CONFIG['colors']
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    
    degrees = sorted(aggregated.keys())
    x_pos = np.array(degrees)
    
    # Extract data
    q_rmse_mean = [aggregated[d]['quantum_rmse_mean'] for d in degrees]
    q_rmse_std = [aggregated[d]['quantum_rmse_std'] for d in degrees]
    c_rmse_mean = [aggregated[d]['classical_rmse_mean'] for d in degrees]
    c_rmse_std = [aggregated[d]['classical_rmse_std'] for d in degrees]
    
    q_corr_mean = [aggregated[d]['quantum_corr_mean'] for d in degrees]
    q_corr_std = [aggregated[d]['quantum_corr_std'] for d in degrees]
    c_corr_mean = [aggregated[d]['classical_corr_mean'] for d in degrees]
    c_corr_std = [aggregated[d]['classical_corr_std'] for d in degrees]
    
    # Panel A: RMSE vs Degree
    width = 0.35
    ax1.bar(x_pos - width/2, c_rmse_mean, width, yerr=c_rmse_std,
           color=colors['classical'], label='Classical NN',
           edgecolor='black', linewidth=0.5, capsize=3)
    ax1.bar(x_pos + width/2, q_rmse_mean, width, yerr=q_rmse_std,
           color=colors['quantum'], label='Quantum (Native)',
           edgecolor='black', linewidth=0.5, capsize=3)
    
    ax1.set_xlabel('Polynomial Degree')
    ax1.set_ylabel('RMSE')
    ax1.set_title('(a) Recovery Error vs Polynomial Degree\n(Native Quantum Arithmetic)')
    ax1.set_xticks(degrees)
    ax1.legend(loc='upper left')
    ax1.set_ylim(0, max(q_rmse_mean) * 1.5 if q_rmse_mean else 0.1)
    ax1.axhline(0.03, color='green', linestyle='--', alpha=0.7, label='PASS threshold')
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Panel B: Correlation vs Degree
    ax2.errorbar(x_pos - 0.1, c_corr_mean, yerr=c_corr_std,
                fmt='s', color=colors['classical'], markersize=8,
                capsize=4, capthick=1.5, label='Classical NN',
                markeredgecolor='black', markeredgewidth=0.5)
    ax2.errorbar(x_pos + 0.1, q_corr_mean, yerr=q_corr_std,
                fmt='o', color=colors['quantum'], markersize=8,
                capsize=4, capthick=1.5, label='Quantum (Native)',
                markeredgecolor='black', markeredgewidth=0.5)
    
    ax2.set_xlabel('Polynomial Degree')
    ax2.set_ylabel('Correlation Coefficient')
    ax2.set_title('(b) Correlation vs Polynomial Degree\n(Native Quantum Arithmetic)')
    ax2.set_xticks(degrees)
    ax2.legend(loc='lower left')
    ax2.set_ylim(0.99, 1.001)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    pdf_path = output_path.rsplit('.', 1)[0] + '.pdf'
    plt.savefig(pdf_path, dpi=300, bbox_inches='tight')
    
    print(f"Figure 1 saved to: {output_path}")
    plt.close()
    return fig


# ==============================================================================
# FIGURE 2: RECOVERY GRID (2x3)
# ==============================================================================

def create_figure2_recovery_grid(aggregated, output_path):
    """Create Figure 2: 2x3 grid showing recovery for each degree (Native Protocol)."""
    configure_publication_style()
    
    colors = PLOT_CONFIG['colors']
    degrees = sorted(aggregated.keys())
    
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
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
        ax.errorbar(x, quantum, yerr=q_err, fmt='o', color=colors['quantum'],
                   markersize=6, capsize=3, capthick=1, elinewidth=1,
                   markeredgecolor='black', markeredgewidth=0.3,
                   label='Quantum (Native)', zorder=3, alpha=0.9)
        
        ax.set_xlabel('$x$')
        ax.set_ylabel('$F(x)$')
        ax.set_title(f'Degree {degree}: {data["name"]}\n(Native Quantum Arithmetic)')
        ax.set_xlim(-1.05, 1.05)
        ax.grid(True, alpha=0.3)
        
        if idx == 0:
            ax.legend(loc='best', fontsize=8)
        
        # Add RMSE annotation
        rmse_text = f'RMSE: {data["quantum_rmse_mean"]:.4f}'
        ax.text(0.95, 0.05, rmse_text, transform=ax.transAxes,
               fontsize=9, ha='right', va='bottom',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    pdf_path = output_path.rsplit('.', 1)[0] + '.pdf'
    plt.savefig(pdf_path, dpi=300, bbox_inches='tight')
    
    print(f"Figure 2 saved to: {output_path}")
    plt.close()
    return fig


# ==============================================================================
# FIGURE 3: ERROR DISTRIBUTION
# ==============================================================================

def create_figure3_error_distribution(aggregated, output_path):
    """Create Figure 3: Error distribution histograms by degree (Native Protocol)."""
    configure_publication_style()
    
    colors = PLOT_CONFIG['colors']
    degrees = sorted(aggregated.keys())
    
    fig, axes = plt.subplots(2, 3, figsize=(12, 7))
    axes = axes.flatten()
    
    for idx, degree in enumerate(degrees):
        ax = axes[idx]
        data = aggregated[degree]
        
        # Collect all errors from all trials
        all_quantum_errors = []
        all_classical_errors = []
        
        for trial_data in data['results']:
            q_errors = trial_data['measured'] - trial_data['theoretical']
            c_errors = trial_data['classical_pred'] - trial_data['theoretical']
            all_quantum_errors.extend(q_errors)
            all_classical_errors.extend(c_errors)
        
        all_quantum_errors = np.array(all_quantum_errors)
        all_classical_errors = np.array(all_classical_errors)
        
        # Histogram
        bins = np.linspace(-0.1, 0.1, 30)
        
        ax.hist(all_classical_errors, bins=bins, color=colors['classical'],
               alpha=0.5, label='Classical', edgecolor='black', linewidth=0.3)
        ax.hist(all_quantum_errors, bins=bins, color=colors['quantum'],
               alpha=0.5, label='Quantum (Native)', edgecolor='black', linewidth=0.3)
        
        # Threshold regions
        ax.axvspan(-0.03, 0.03, color=colors['pass_region'], alpha=0.3)
        ax.axvline(0, color='black', linestyle='-', linewidth=1)
        
        ax.set_xlabel('Recovery Error')
        ax.set_ylabel('Count')
        ax.set_title(f'Degree {degree}: {data["name"]}\n(Native Quantum Arithmetic)')
        ax.set_xlim(-0.1, 0.1)
        
        if idx == 0:
            ax.legend(loc='upper right', fontsize=8)
        
        # Add pass rate
        pass_text = f'Pass: {data["pass_rate_mean"]*100:.0f}%'
        ax.text(0.95, 0.95, pass_text, transform=ax.transAxes,
               fontsize=9, ha='right', va='top',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    pdf_path = output_path.rsplit('.', 1)[0] + '.pdf'
    plt.savefig(pdf_path, dpi=300, bbox_inches='tight')
    
    print(f"Figure 3 saved to: {output_path}")
    plt.close()
    return fig


# ==============================================================================
# FIGURE 4: CORRELATION SUMMARY
# ==============================================================================

def create_figure4_correlation_summary(aggregated, all_data, output_path):
    """Create Figure 4: Correlation analysis scatter plots (Native Protocol)."""
    configure_publication_style()
    
    colors = PLOT_CONFIG['colors']
    degrees = sorted(aggregated.keys())
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5))
    
    # Color map for degrees
    degree_colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(degrees)))
    
    # Panel A: All points colored by degree
    for idx, degree in enumerate(degrees):
        data = aggregated[degree]
        
        for trial_data in data['results']:
            theo = trial_data['theoretical']
            quantum = trial_data['measured']
            
            ax1.scatter(theo, quantum, c=[degree_colors[idx]], s=20, alpha=0.6,
                       label=f'Deg {degree}' if trial_data == data['results'][0] else '')
    
    # Ideal line
    ax1.plot([-0.6, 0.6], [-0.6, 0.6], 'k--', linewidth=1.5, label='Ideal')
    
    ax1.set_xlabel('Theoretical Value')
    ax1.set_ylabel('Quantum Measured Value (Native)')
    ax1.set_title('(a) All Measurements vs Theory\n(Native Quantum Arithmetic)')
    ax1.set_aspect('equal')
    ax1.set_xlim(-0.6, 0.6)
    ax1.set_ylim(-0.6, 0.6)
    ax1.legend(loc='lower right', fontsize=8, ncol=2)
    ax1.grid(True, alpha=0.3)
    
    # Panel B: Correlation and pass rate by degree
    pass_rates = [aggregated[d]['pass_rate_mean'] * 100 for d in degrees]
    q_corr = [aggregated[d]['quantum_corr_mean'] for d in degrees]
    
    ax2_twin = ax2.twinx()
    
    bars = ax2.bar(degrees, pass_rates, color=colors['quantum'], alpha=0.7,
                  edgecolor='black', linewidth=0.5, label='Pass Rate')
    ax2.set_ylabel('Pass Rate (%)', color=colors['quantum'])
    ax2.set_ylim(0, 110)
    ax2.tick_params(axis='y', labelcolor=colors['quantum'])
    
    ax2_twin.plot(degrees, q_corr, 'o-', color=colors['theoretical'],
                 linewidth=2, markersize=8, label='Correlation')
    ax2_twin.set_ylabel('Correlation', color=colors['theoretical'])
    ax2_twin.set_ylim(0.99, 1.001)
    ax2_twin.tick_params(axis='y', labelcolor=colors['theoretical'])
    
    ax2.set_xlabel('Polynomial Degree')
    ax2.set_title('(b) Performance Metrics by Degree\n(Native Quantum Arithmetic)')
    ax2.set_xticks(degrees)
    
    # Combined legend
    lines1, labels1 = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2_twin.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc='lower left')
    
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    pdf_path = output_path.rsplit('.', 1)[0] + '.pdf'
    plt.savefig(pdf_path, dpi=300, bbox_inches='tight')
    
    print(f"Figure 4 saved to: {output_path}")
    plt.close()
    return fig


# ==============================================================================
# LATEX TABLES
# ==============================================================================

def generate_latex_table(aggregated, output_path):
    """Generate LaTeX table of results for native approach."""
    degrees = sorted(aggregated.keys())
    
    latex = r"""
\begin{table}[ht]
\centering
\caption{Polynomial Recovery Accuracy: Native Quantum Arithmetic Protocol}
\label{tab:results-native}
\begin{tabular}{lccccc}
\toprule
Degree & Name & Q-RMSE & C-RMSE & Correlation & Pass Rate \\
\midrule
"""
    
    for degree in degrees:
        data = aggregated[degree]
        q_rmse = f"${data['quantum_rmse_mean']:.4f} \\pm {data['quantum_rmse_std']:.4f}$"
        c_rmse = f"${data['classical_rmse_mean']:.4f} \\pm {data['classical_rmse_std']:.4f}$"
        corr = f"${data['quantum_corr_mean']:.4f}$"
        pass_rate = f"${data['pass_rate_mean']*100:.1f}\\%$"
        
        latex += f"{degree} & {data['name']} & {q_rmse} & {c_rmse} & {corr} & {pass_rate} \\\\\n"
    
    latex += r"""\bottomrule
\end{tabular}
\end{table}
"""
    
    with open(output_path, 'w') as f:
        f.write(latex)
    
    print(f"LaTeX table saved to: {output_path}")
    return latex


# ==============================================================================
# MAIN
# ==============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description='Generate publication figures for native polynomial protocol'
    )
    parser.add_argument('--input', default='results/cloud',
                       help='Input directory with H5 result files')
    parser.add_argument('--output', default='figures/paper/native',
                       help='Output directory for figures')
    parser.add_argument('--all', action='store_true',
                       help='Generate all figures')
    parser.add_argument('--fig1', action='store_true',
                       help='Generate Figure 1: Degree scaling')
    parser.add_argument('--fig2', action='store_true',
                       help='Generate Figure 2: Recovery grid')
    parser.add_argument('--fig3', action='store_true',
                       help='Generate Figure 3: Error distribution')
    parser.add_argument('--fig4', action='store_true',
                       help='Generate Figure 4: Correlation summary')
    parser.add_argument('--tables', action='store_true',
                       help='Generate LaTeX tables')
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
    
    # Load data (filtered for native approach)
    all_data = load_native_results(input_dir)
    if not all_data:
        print("No native approach data found!")
        sys.exit(1)
    
    aggregated = aggregate_by_degree(all_data)
    
    # Determine which figures to generate
    generate_all = args.all or not (args.fig1 or args.fig2 or args.fig3 or args.fig4 or args.tables)
    
    print("\n" + "=" * 60)
    print("GENERATING NATIVE POLYNOMIAL PROTOCOL FIGURES")
    print("=" * 60)
    
    if generate_all or args.fig1:
        fig1_path = os.path.join(output_dir, 'fig1_degree_scaling.png')
        create_figure1_degree_scaling(aggregated, fig1_path)
    
    if generate_all or args.fig2:
        fig2_path = os.path.join(output_dir, 'fig2_recovery_grid.png')
        create_figure2_recovery_grid(aggregated, fig2_path)
    
    if generate_all or args.fig3:
        fig3_path = os.path.join(output_dir, 'fig3_error_distribution.png')
        create_figure3_error_distribution(aggregated, fig3_path)
    
    if generate_all or args.fig4:
        fig4_path = os.path.join(output_dir, 'fig4_correlation_summary.png')
        create_figure4_correlation_summary(aggregated, all_data, fig4_path)
    
    if generate_all or args.tables:
        table_path = os.path.join(output_dir, 'table_results.tex')
        generate_latex_table(aggregated, table_path)
    
    print("\n" + "=" * 60)
    print(f"All outputs saved to: {output_dir}")
    print("=" * 60)
    
    if not args.no_display:
        plt.show()
