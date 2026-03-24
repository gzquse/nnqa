
#!/usr/bin/env python3
"""
IBM Cloud Results Plotter
=========================

Generates publication-quality figures from IBM Cloud native polynomial experiments.
Adapted from plot_ionq_results.py but for IBM result format.

Usage:
    python plot_ibm_results.py --input results/ibm_miami --output figures/ibm_miami
"""

import sys
import os
import numpy as np
import glob
import argparse
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# Add parent paths
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from toolbox.Util_H5io4 import read4_data_hdf5
from research_config import POLYNOMIALS, PLOT_CONFIG, evaluate_polynomial

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

def load_ibm_results(input_dir):
    """Load all H5 result files from IBM Cloud experiments."""
    h5_files = glob.glob(os.path.join(input_dir, 'deg*.h5'))
    # Exclude submitted files
    h5_files = [f for f in h5_files if '_submitted' not in f]
    
    all_data = []
    for h5_file in sorted(h5_files):
        try:
            data, meta = read4_data_hdf5(h5_file, verb=0)
            
            # Use metadata metrics if available, otherwise compute them
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
                'backend': meta.get('backend', 'ibm_unknown'),
                'provider': 'ibm_cloud',
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
    
    print(f"Loaded {len(all_data)} IBM result files from {input_dir}")
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

def create_figure1_degree_scaling(aggregated, output_path, backend_name="IBM Quantum"):
    """Create Figure 1: RMSE vs polynomial degree with error bars."""
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
           color=colors['quantum'], label=f'{backend_name} Quantum',
           edgecolor='black', linewidth=0.5, capsize=3)
    
    ax1.set_xlabel('Polynomial Degree')
    ax1.set_ylabel('RMSE')
    ax1.set_title(f'(a) Recovery Error vs Polynomial Degree\n({backend_name})')
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
                capsize=4, capthick=1.5, label=f'{backend_name} Quantum',
                markeredgecolor='black', markeredgewidth=0.5)
    
    ax2.set_xlabel('Polynomial Degree')
    ax2.set_ylabel('Correlation Coefficient')
    ax2.set_title(f'(b) Correlation vs Polynomial Degree\n({backend_name})')
    ax2.set_xticks(degrees)
    ax2.legend(loc='lower left')
    ax2.set_ylim(0.9, 1.01)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    pdf_path = output_path.rsplit('.', 1)[0] + '.pdf'
    plt.savefig(pdf_path, dpi=300, bbox_inches='tight')
    
    print(f"Figure 1 saved to: {output_path}")
    plt.close()
    return fig

# ==============================================================================
# FIGURE 2: RECOVERY GRID
# ==============================================================================

def create_figure2_recovery_grid(aggregated, output_path, backend_name="IBM Quantum"):
    """Create Figure 2: Grid showing recovery for each degree."""
    configure_publication_style()
    
    colors = PLOT_CONFIG['colors']
    degrees = sorted(aggregated.keys())
    
    # Determine grid size
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
        ax.errorbar(x, quantum, yerr=q_err, fmt='o', color=colors['quantum'],
                   markersize=6, capsize=3, capthick=1, elinewidth=1,
                   markeredgecolor='black', markeredgewidth=0.3,
                   label=f'{backend_name}', zorder=3, alpha=0.9)
        
        ax.set_xlabel('$x$')
        ax.set_ylabel('$F(x)$')
        ax.set_title(f'Degree {degree}: {data["name"]}\n({backend_name})')
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
# MAIN
# ==============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description='Generate figures for IBM Cloud results'
    )
    parser.add_argument('--input', required=True,
                       help='Input directory with H5 result files')
    parser.add_argument('--output', required=True,
                       help='Output directory for figures')
    parser.add_argument('--backend-name', default='IBM Quantum',
                       help='Name of backend to display in plots')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    
    os.makedirs(args.output, exist_ok=True)
    
    all_data = load_ibm_results(args.input)
    if not all_data:
        print("No results found!")
        sys.exit(1)
        
    aggregated = aggregate_by_degree(all_data)
    
    print(f"Generating plots for {len(aggregated)} degrees...")
    
    create_figure1_degree_scaling(aggregated, 
                                 os.path.join(args.output, 'fig1_degree_scaling.png'),
                                 args.backend_name)
    
    create_figure2_recovery_grid(aggregated, 
                                os.path.join(args.output, 'fig2_recovery_grid.png'),
                                args.backend_name)
    
    print("Done!")
