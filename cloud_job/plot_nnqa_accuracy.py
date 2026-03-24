#!/usr/bin/env python3
"""
Plot NNQA accuracy results with error bars

Generates publication-quality plots showing:
- Theoretical vs measured correlation
- Recovery error histogram
- Error bar analysis

Usage:
    ./plot_nnqa_accuracy.py --basePath cloud_job/out --expName boston_abc123
    ./plot_nnqa_accuracy.py --basePath cloud_job/out --expName boston_abc123 -p a b
"""

import sys
import os
import numpy as np
from pprint import pprint
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from toolbox.Util_H5io4 import read4_data_hdf5, write4_data_hdf5

import argparse


# ==============================================================================
# MATPLOTLIB CONFIGURATION
# ==============================================================================

def configure_matplotlib():
    """Configure matplotlib for publication-quality figures."""
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'DejaVu Serif'],
        'font.size': 11,
        'axes.labelsize': 12,
        'axes.titlesize': 13,
        'legend.fontsize': 10,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'figure.figsize': (12, 5),
        'figure.dpi': 150,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'axes.linewidth': 0.8,
        'axes.grid': True,
        'grid.alpha': 0.3,
        'lines.linewidth': 2.0,
        'lines.markersize': 6,
    })


# ==============================================================================
# PLOTTING FUNCTIONS
# ==============================================================================

def compute_correlation(x, y):
    """Compute Pearson correlation coefficient."""
    return np.corrcoef(x, y)[0, 1]


def plot_correlation(ax, theoretical, measured, measured_err):
    """Plot theoretical vs measured correlation with error bars."""
    
    # Scatter with error bars
    ax.errorbar(theoretical, measured, yerr=measured_err,
                fmt='o', color='#648FFF', markersize=6,
                capsize=3, capthick=1.5, elinewidth=1.5,
                markeredgecolor='black', markeredgewidth=0.5,
                alpha=0.8, label='Measured')
    
    # Ideal line
    lim_min = min(theoretical.min(), measured.min()) - 0.1
    lim_max = max(theoretical.max(), measured.max()) + 0.1
    ax.plot([lim_min, lim_max], [lim_min, lim_max], 'k--', lw=1.0, label='Ideal')
    
    # Correlation line
    corr = compute_correlation(theoretical, measured)
    m = corr * np.std(measured) / np.std(theoretical)
    c = np.mean(measured) - m * np.mean(theoretical)
    x_line = np.array([lim_min, lim_max])
    y_line = m * x_line + c
    ax.plot(x_line, y_line, 'r--', lw=1.0, alpha=0.7)
    
    ax.text(0.05, 0.92, f'Correlation: {corr:.4f}',
           transform=ax.transAxes, color='r', fontsize=10)
    
    ax.set_xlim(lim_min, lim_max)
    ax.set_ylim(lim_min, lim_max)
    ax.set_aspect('equal')
    ax.set_xlabel('Theoretical Value')
    ax.set_ylabel('Measured Value')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)


def plot_error_histogram(ax, residuals, measured_err):
    """Plot error histogram with statistics."""
    
    # Histogram
    ax.hist(residuals, bins=20, color='#DC267F', alpha=0.7, edgecolor='black')
    
    # Statistics
    mean = np.mean(residuals)
    std = np.std(residuals)
    rmse = np.sqrt(np.mean(residuals**2))
    N = len(residuals)
    se_s = std / np.sqrt(2 * (N - 1))
    
    ax.axvline(mean, color='r', linestyle='--', linewidth=2)
    ax.axvline(0, color='k', linestyle='-', linewidth=1)
    
    stats_text = (f'Mean: {mean:.4f}\n'
                 f'RMSE: {rmse:.4f} +/- {se_s:.4f}\n'
                 f'N: {N}')
    ax.text(0.95, 0.95, stats_text, transform=ax.transAxes,
           verticalalignment='top', horizontalalignment='right',
           fontsize=10, color='r',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    ax.set_xlabel('Recovery Error (Theoretical - Measured)')
    ax.set_ylabel('Count')
    ax.xaxis.set_major_locator(ticker.MaxNLocator(5))


def plot_error_vs_value(ax, theoretical, residuals, measured_err):
    """Plot error vs theoretical value."""
    
    ax.errorbar(theoretical, residuals, yerr=measured_err,
                fmt='s', color='#785EF0', markersize=5,
                capsize=3, capthick=1, elinewidth=1,
                markeredgecolor='black', markeredgewidth=0.5,
                alpha=0.8)
    
    # Threshold regions
    ax.axhspan(-0.03, 0.03, color='#90EE90', alpha=0.3, label='PASS')
    ax.axhspan(-0.1, -0.03, color='#FFE4B5', alpha=0.3)
    ax.axhspan(0.03, 0.1, color='#FFE4B5', alpha=0.3, label='POOR')
    ax.axhline(0, color='k', linestyle='-', linewidth=1)
    
    # Correlation
    corr = compute_correlation(theoretical, residuals)
    ax.text(0.05, 0.92, f'Residual Corr: {corr:.4f}',
           transform=ax.transAxes, color='r', fontsize=10)
    
    ax.set_xlabel('Theoretical Value')
    ax.set_ylabel('Recovery Error')
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True, alpha=0.3)


def plot_accuracy_summary(expD, expMD, save_path=None):
    """Create complete accuracy summary plot."""
    configure_matplotlib()
    
    theoretical = expD['theoretical']
    measured = expD['measured']
    measured_err = expD['measured_err']
    residuals = theoretical - measured
    
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    
    # Panel A: Correlation
    plot_correlation(axes[0], theoretical, measured, measured_err)
    axes[0].set_title(f"(a) {expMD['short_name']}: Correlation")
    
    # Panel B: Error histogram
    plot_error_histogram(axes[1], residuals, measured_err)
    axes[1].set_title(f"(b) {expMD['submit']['backend']}: Error Distribution")
    
    # Panel C: Error vs value
    plot_error_vs_value(axes[2], theoretical, residuals, measured_err)
    axes[2].set_title('(c) Recovery Error Analysis')
    
    # Add metadata
    meta_text = (f"Backend: {expMD['submit']['backend']}\n"
                f"Shots: {expMD['submit']['num_shots']}\n"
                f"Samples: {expMD['payload']['num_sample']}\n"
                f"RC: {expMD['submit']['random_compilation']}\n"
                f"DD: {expMD['submit']['dynamical_decoupling']}")
    fig.text(0.99, 0.02, meta_text, fontsize=8, ha='right', va='bottom',
            transform=fig.transFigure,
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to: {save_path}")
    
    return fig


def plot_multi_experiment_comparison(experiments, save_path=None):
    """Compare multiple experiments in one plot."""
    configure_matplotlib()
    
    n_exp = len(experiments)
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, n_exp))
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    rmse_values = []
    labels = []
    
    for i, (name, expD, expMD) in enumerate(experiments):
        theoretical = expD['theoretical']
        measured = expD['measured']
        residuals = theoretical - measured
        
        rmse = np.sqrt(np.mean(residuals**2))
        rmse_values.append(rmse)
        labels.append(name)
        
        # Scatter plot
        ax1.scatter(theoretical, measured, c=[colors[i]], s=30, alpha=0.7,
                   label=f'{name} (RMSE={rmse:.4f})')
    
    # Ideal line
    ax1.plot([-1, 1], [-1, 1], 'k--', lw=1)
    ax1.set_xlabel('Theoretical Value')
    ax1.set_ylabel('Measured Value')
    ax1.set_title('Multi-Experiment Correlation')
    ax1.legend(loc='lower right', fontsize=8)
    ax1.set_aspect('equal')
    ax1.grid(True, alpha=0.3)
    
    # Bar chart of RMSE
    bars = ax2.bar(range(n_exp), rmse_values, color=colors, edgecolor='black')
    ax2.set_xticks(range(n_exp))
    ax2.set_xticklabels(labels, rotation=45, ha='right')
    ax2.set_ylabel('RMSE')
    ax2.set_title('Recovery Error Comparison')
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Add values on bars
    for bar, val in zip(bars, rmse_values):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
                f'{val:.4f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to: {save_path}")
    
    return fig


# ==============================================================================
# POSTPROCESSING
# ==============================================================================

def postprocess_results(expD, expMD):
    """Compute postprocessing metrics."""
    theoretical = expD['theoretical']
    measured = expD['measured']
    residuals = theoretical - measured
    
    pom = expMD['postproc']
    pom['res_mean'] = float(np.mean(residuals))
    pom['res_std'] = float(np.std(residuals))
    pom['rmse'] = float(np.sqrt(np.mean(residuals**2)))
    pom['correlation'] = float(compute_correlation(theoretical, measured))
    
    N = len(residuals)
    pom['res_SE_s'] = float(pom['res_std'] / np.sqrt(2 * (N - 1)))
    
    # Count pass/poor/fail
    abs_residuals = np.abs(residuals)
    pom['pass_rate'] = float(np.mean(abs_residuals < 0.03))
    pom['poor_rate'] = float(np.mean((abs_residuals >= 0.03) & (abs_residuals < 0.1)))
    pom['fail_rate'] = float(np.mean(abs_residuals >= 0.1))
    
    return pom


# ==============================================================================
# ARGUMENT PARSER
# ==============================================================================

def get_parser():
    parser = argparse.ArgumentParser(
        description='Plot NNQA accuracy results'
    )
    parser.add_argument("-v", "--verb", type=int, default=1,
                       help="verbosity level")
    parser.add_argument("--basePath", default='cloud_job/out',
                       help="base directory")
    parser.add_argument('-e', "--expName", nargs='+', required=True,
                       help='experiment name(s)')
    parser.add_argument("-p", "--showPlots", default='a', nargs='+',
                       help="plots to show: a=summary, b=comparison")
    parser.add_argument("-Y", "--noXterm", action='store_false', default=True,
                       help="disable interactive display")
    parser.add_argument("--format", default='pdf', choices=['pdf', 'png', 'svg'],
                       help="output format")
    
    args = parser.parse_args()
    
    args.dataPath = os.path.join(args.basePath, 'meas')
    args.outPath = os.path.join(args.basePath, 'plots')
    args.showPlots = ''.join(args.showPlots)
    
    for arg in vars(args):
        print(f'myArgs: {arg} = {getattr(args, arg)}')
    
    return args


# ==============================================================================
# MAIN
# ==============================================================================

if __name__ == "__main__":
    args = get_parser()
    
    os.makedirs(args.outPath, exist_ok=True)
    
    experiments = []
    
    for expName in args.expName:
        inpF = expName + '.meas.h5'
        expD, expMD = read4_data_hdf5(os.path.join(args.dataPath, inpF))
        
        # Postprocess
        pom = postprocess_results(expD, expMD)
        
        print(f"\n{'='*60}")
        print(f"Experiment: {expName}")
        print(f"{'='*60}")
        print(f"Backend: {expMD['submit']['backend']}")
        print(f"Samples: {expMD['payload']['num_sample']}")
        print(f"Shots: {expMD['submit']['num_shots']}")
        print(f"RMSE: {pom['rmse']:.4f} +/- {pom['res_SE_s']:.4f}")
        print(f"Correlation: {pom['correlation']:.4f}")
        print(f"Pass Rate: {pom['pass_rate']*100:.1f}%")
        print(f"Poor Rate: {pom['poor_rate']*100:.1f}%")
        print(f"Fail Rate: {pom['fail_rate']*100:.1f}%")
        
        experiments.append((expName, expD, expMD))
        
        # Save updated metadata
        postPath = os.path.join(args.basePath, 'post')
        os.makedirs(postPath, exist_ok=True)
        outF = os.path.join(postPath, expMD['short_name'] + '.h5')
        write4_data_hdf5(expD, outF, expMD)
    
    # Plot individual summaries
    if 'a' in args.showPlots:
        for expName, expD, expMD in experiments:
            save_path = os.path.join(args.outPath, 
                                    f"{expMD['short_name']}_accuracy.{args.format}")
            fig = plot_accuracy_summary(expD, expMD, save_path)
    
    # Plot comparison if multiple experiments
    if 'b' in args.showPlots and len(experiments) > 1:
        save_path = os.path.join(args.outPath, f"comparison.{args.format}")
        fig = plot_multi_experiment_comparison(experiments, save_path)
    
    if args.noXterm:
        plt.show()
    
    print('\nDone!')

