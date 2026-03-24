#!/usr/bin/env python3
"""
Single-Column Publication Figures for RevTeX
=============================================

Generates high-quality, concise figures optimized for single-column layout
in RevTeX LaTeX format. Figures are sized for ~3.3 inch (84mm) column width
with appropriate font sizes for publication.

Usage:
    python plot_single_column.py --input results/cloud --output figures/paper/single_column_cloud
    python plot_single_column.py --all
"""

import sys
import os
import numpy as np
import json
import glob
import argparse
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as ticker

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from research_config import POLYNOMIALS, PLOT_CONFIG, format_polynomial_latex
from toolbox.Util_H5io4 import read4_data_hdf5


# ==============================================================================
# SINGLE-COLUMN STYLE CONFIGURATION (RevTeX Compatible)
# ==============================================================================

# RevTeX single column width: ~3.3 inches (84mm)
SINGLE_COL_WIDTH = 3.3  # inches
DOUBLE_COL_WIDTH = 6.7  # inches

def configure_single_column_style():
    """Configure matplotlib for single-column RevTeX publication figures."""
    plt.rcParams.update({
        # Font settings for RevTeX
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'Computer Modern Roman', 'DejaVu Serif'],
        'font.size': 10,          # Base font size (RevTeX compatible)
        'axes.labelsize': 10,     # Axis labels
        'axes.titlesize': 10,     # Subplot titles
        'figure.titlesize': 11,   # Figure title (if used)
        'legend.fontsize': 9,     # Legend
        'xtick.labelsize': 9,     # X-axis ticks
        'ytick.labelsize': 9,     # Y-axis ticks
        'text.usetex': False,     # Can set to True if LaTeX is available
        
        # Figure quality
        'figure.dpi': 150,
        'savefig.dpi': 300,       # High quality output
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.02,  # Minimal padding
        
        # Axis styling
        'axes.linewidth': 0.8,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'grid.alpha': 0.3,
        'grid.linewidth': 0.5,
        'grid.linestyle': '--',
        
        # Line and marker styling
        'lines.linewidth': 1.5,
        'lines.markersize': 4,
        'lines.markeredgewidth': 0.5,
        
        # Legend styling
        'legend.framealpha': 0.95,
        'legend.edgecolor': '0.6',
        'legend.fancybox': False,
        'legend.frameon': True,
        'legend.numpoints': 1,
        'legend.scatterpoints': 1,
        
        # Mathtext settings
        'mathtext.default': 'regular',
        'mathtext.fontset': 'stix',
    })


# ==============================================================================
# DATA LOADING (Same as plot_native_polynomial.py)
# ==============================================================================

def load_ionq_stress_test_results(input_dir):
    """Load IonQ stress test results (degrees 1-35)."""
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
                'polynomial_name': meta.get('polynomial_name', 'Degree ' + str(meta['degree'])),
                'metrics': meta['metrics'],
                'config': meta.get('config', {}),
                'x_values': data['x_values'],
                'theoretical': data['theoretical'],
                'classical_pred': data['classical_pred'],
                'measured': data['measured'],
                'measured_err': data.get('measured_err', np.zeros_like(data['measured'])),
            }
            all_data.append(result)
        except Exception as e:
            print("Warning: Skipping " + str(h5_file) + " due to error: " + str(e))
            continue
    
    print("Loaded " + str(len(all_data)) + " IonQ stress test result files from " + input_dir)
    return all_data


def load_native_results(input_dir, filter_approach='native'):
    """Load all H5 result files from directory, optionally filtering by approach."""
    # input_dir should already be resolved from main(), just use it as-is
    h5_files = glob.glob(os.path.join(input_dir, 'deg*.h5'))
    h5_files = [f for f in h5_files if '_submitted' not in f]
    
    all_data = []
    for h5_file in sorted(h5_files):
        try:
            data, meta = read4_data_hdf5(h5_file, verb=0)
            
            approach = meta.get('approach', None)
            # If approach not set, infer from directory or assume based on context
            if approach is None:
                if 'direct' in input_dir.lower():
                    approach = 'direct'
                elif 'cloud' in input_dir.lower() or 'native' in input_dir.lower():
                    approach = 'native'
                else:
                    approach = filter_approach  # Default to requested filter
            
            # Filter by approach if specified
            if filter_approach and approach != filter_approach:
                continue
            
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
                'polynomial_name': meta.get('polynomial_name', 'Degree ' + str(meta['degree'])),
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
            print("Warning: Skipping " + str(h5_file) + " due to error: " + str(e))
            continue
    
    approach_label = filter_approach if filter_approach else 'all approaches'
    print("Loaded " + str(len(all_data)) + " " + approach_label + " result files from " + input_dir)
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
        
        # Get polynomial name: use POLYNOMIALS if available, otherwise use from result or default
        if degree in POLYNOMIALS:
            poly_name = POLYNOMIALS[degree]['name']
        elif degree_results and 'polynomial_name' in degree_results[0]:
            poly_name = degree_results[0]['polynomial_name']
        else:
            poly_name = f'Degree {degree}'
        
        aggregated[degree] = {
            'name': poly_name,
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
# FIGURE 1: DEGREE SCALING (Stacked vertically for single column)
# ==============================================================================

def create_figure1_degree_scaling(aggregated, output_path):
    """Create Figure 1: RMSE and Correlation vs polynomial degree (template style)."""
    # RC Params matching template
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'DejaVu Serif'],
        'font.size': 8,
        'axes.labelsize': 8,
        'axes.titlesize': 8,
        'legend.fontsize': 7,
        'xtick.labelsize': 7,
        'ytick.labelsize': 7,
        'lines.linewidth': 1.0,
        'axes.linewidth': 0.8,
        'figure.dpi': 300,
        'axes.spines.top': False,
        'axes.spines.right': False,
    })
    
    colors = PLOT_CONFIG['colors']
    degrees = sorted(aggregated.keys())
    degrees_array = np.array(degrees)
    
    # Extract data
    q_rmse_mean = np.array([aggregated[d]['quantum_rmse_mean'] for d in degrees])
    q_rmse_std = np.array([aggregated[d]['quantum_rmse_std'] for d in degrees])
    q_corr_mean = np.array([aggregated[d]['quantum_corr_mean'] for d in degrees])
    q_corr_std = np.array([aggregated[d]['quantum_corr_std'] for d in degrees])
    
    # Colors matching template
    color_rmse = colors['quantum']  # '#DC267F' - Magenta
    color_corr = colors['classical']  # '#648FFF' - Blue
    
    # Figure Layout (Manual GridSpec for Equal Widths)
    fig = plt.figure(figsize=(4.5, 2.3))
    
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 1],
                          left=0.14, right=0.97,  
                          bottom=0.16, top=0.90,
                          wspace=0.30)           
    
    # Left Plot: Broken Axis for RMSE
    gs_left = gs[0].subgridspec(2, 1, height_ratios=[5, 1], hspace=0.15)
    ax1_top = fig.add_subplot(gs_left[0])
    ax1_bottom = fig.add_subplot(gs_left[1])
    
    # Right Plot: Correlation (top) + Distribution (bottom), equal heights
    gs_right = gs[1].subgridspec(2, 1, height_ratios=[1, 1], hspace=0.25)
    ax2 = fig.add_subplot(gs_right[0])      # correlation
    ax2_hist = fig.add_subplot(gs_right[1]) # histogram
    
    # Calculate RMSE mean for reference line
    rmse_mean = np.mean(q_rmse_mean)
    
    # Filter data for fill_between (points near mean)
    mask = np.abs(q_rmse_mean - rmse_mean) <= 0.04
    degrees_filt = degrees_array[mask]
    rmse_filt = q_rmse_mean[mask]
    rmse_err_filt = q_rmse_std[mask]
    
    # ==========================================
    # Plot 1: RMSE (Broken Axis)
    # ==========================================
    for ax in [ax1_top, ax1_bottom]:
        ax.axhline(rmse_mean, color='gray', linestyle='--', linewidth=0.8, alpha=0.7, zorder=1)
        if len(degrees_filt) > 0:
            ax.fill_between(degrees_filt, 0, rmse_filt, color=color_rmse, alpha=0.15, linewidth=0, zorder=1)
        
        ax.errorbar(degrees_array, q_rmse_mean, yerr=q_rmse_std, fmt='none', 
                     ecolor=color_rmse, elinewidth=1.2, capsize=1.5, zorder=2)
        ax.scatter(degrees_array, q_rmse_mean, s=15, facecolors=color_rmse, edgecolors=color_rmse, 
                    linewidths=1.0, zorder=3)
    
    # Configure Breaks
    # Top part: show main range (adjust based on data)
    rmse_min = np.min(q_rmse_mean)
    rmse_max = np.max(q_rmse_mean)
    rmse_range = rmse_max - rmse_min
    top_ylim_low = max(rmse_min - 0.01, 0.0)
    top_ylim_high = rmse_max + 0.01
    
    ax1_top.set_ylim(top_ylim_low, top_ylim_high) 
    ax1_top.set_xlim(0, max(degrees_array) + 1)
    ax1_top.spines['bottom'].set_visible(False)
    ax1_top.tick_params(labelbottom=False, bottom=False)
    
    # Ticks: 0.01 interval if range is small, otherwise auto
    if rmse_range < 0.1:
        ax1_top.yaxis.set_major_locator(ticker.MultipleLocator(0.01)) 
        ax1_top.yaxis.set_minor_locator(ticker.MultipleLocator(0.005))
    
    # Bottom part: show small values near zero
    ax1_bottom.set_ylim(0, 0.02)
    ax1_bottom.set_xlim(0, max(degrees_array) + 1)
    ax1_bottom.spines['top'].set_visible(False)
    ax1_bottom.set_yticks([0])
    ax1_bottom.set_xlabel('Polynomial Degree')
    
    # Diagonal break lines
    d = .015 
    kwargs = dict(transform=ax1_top.transAxes, color='k', clip_on=False, linewidth=0.8)
    ax1_top.plot((-d, +d), (-d, +d), **kwargs)        
    kwargs.update(transform=ax1_bottom.transAxes)  
    ax1_bottom.plot((-d, +d), (1 - d, 1 + d), **kwargs)  
    
    ax1_top.set_ylabel('RMSE', labelpad=5) 
    ax1_top.set_title('(a) RMSE')
    
    # ==========================================
    # Plot 2: Correlation
    # ==========================================
    # Reference line at 1.0
    ax2.plot([0, max(degrees_array) + 1], [1.0, 1.0], '--', color='gray', 
             linewidth=0.8, alpha=0.7, label='Classical')
    
    # Linear fit for predicted trend
    if len(degrees_array) > 1:
        z = np.polyfit(degrees_array, q_corr_mean, 1)
        p = np.poly1d(z)
        x_fit = np.linspace(0, max(degrees_array) + 1, 100)
        ax2.plot(x_fit, p(x_fit), linestyle='--', color='black', linewidth=0.8, 
                label='Predicted', alpha=0.7, zorder=2)
    
    ax2.errorbar(degrees_array, q_corr_mean, yerr=q_corr_std, fmt='none', 
                 ecolor=color_corr, elinewidth=0.8, capsize=1.0, zorder=2)
    ax2.scatter(degrees_array, q_corr_mean, s=15, facecolors='none', edgecolors=color_corr, 
                linewidths=0.8, zorder=3, label='Correlation')
    ax2.set_xlabel('Polynomial Degree')
    ax2.set_ylabel('Correlation', labelpad=2)
    ax2.set_xlim(0, max(degrees_array) + 1)
    ax2.set_ylim(0.985, 1.001) 
    ax2.set_title('(b) Correlation', pad=2)
    
    # Histogram for correlation distribution (bottom panel)
    # Collect all correlation values from all trials
    all_corr_values = []
    for degree in degrees:
        for trial_data in aggregated[degree]['results']:
            theo = trial_data['theoretical']
            quantum = trial_data['measured']
            corr = np.corrcoef(theo, quantum)[0, 1]
            if not np.isnan(corr):
                all_corr_values.append(corr)
    
    if len(all_corr_values) > 0:
        ax2_hist.hist(
            all_corr_values,
            bins=15,
            color=color_corr,
            alpha=0.5,
            density=True,
            edgecolor='none',
        )
        ax2_hist.set_xlabel('Correlation', labelpad=2)
        ax2_hist.set_yticks([])
        ax2_hist.tick_params(labelsize=7)
        ax2_hist.set_xlim(0.985, 1.001)
        ax2_hist.spines['top'].set_visible(False)
        ax2_hist.spines['right'].set_visible(False)
        ax2_hist.spines['left'].set_visible(False)
        ax2_hist.set_title('Distribution', pad=2)
    else:
        ax2_hist.set_visible(False)
    
    # Save
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    pdf_path = output_path.rsplit('.', 1)[0] + '.pdf'
    plt.savefig(pdf_path, dpi=300, bbox_inches='tight')
    
    print("Figure 1 saved to: " + output_path)
    plt.close()


# ==============================================================================
# FIGURE 2: RECOVERY GRID (3x2 for single column)
# ==============================================================================

def create_figure2_recovery_grid(aggregated, output_path):
    """Create Figure 2: 3x2 grid showing recovery for each degree (single-column)."""
    configure_single_column_style()
    
    colors = PLOT_CONFIG['colors']
    degrees = sorted(aggregated.keys())
    
    # 3 columns, 2 rows for single column layout
    # Reduced height for more compact ICML single column figure
    fig, axes = plt.subplots(3, 2, figsize=(SINGLE_COL_WIDTH, 4.8))
    axes = axes.flatten()
    
    for idx, degree in enumerate(degrees):
        ax = axes[idx]
        data = aggregated[degree]
        
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
        
        # Plot theoretical curve (dashed, above all dots)
        ax.plot(
            x_dense,
            y_dense,
            color=colors['theoretical'],
            linewidth=1.5,
            linestyle='--',
            label='Theory',
            zorder=10,
        )
        
        # Calculate residuals
        classical_residuals = classical - theo
        quantum_residuals = quantum - theo
        
        # Define colors for paper (light blue vs orange)
        classical_color = '#8EC7FF'  # Light blue
        quantum_color = '#FE6100'    # Orange
        
        # Plot classical points (no outline/frame)
        ax.scatter(x, classical, color=classical_color, s=25, marker='s',
                  edgecolors='none', linewidth=0.0, label='Classical', zorder=2, alpha=0.9)
        
        # Plot quantum points (solid triangle, slightly smaller than squares)
        ax.scatter(x, quantum, color=quantum_color, s=20, marker='^',
                  edgecolors='none', linewidth=0.0,
                  label='Quantum', zorder=3, alpha=0.9)
        
        # Set axis limits and labels
        ax.set_xlabel('$x$', fontsize=10)
        ax.set_ylabel('$F(x)$', fontsize=10)
        ax.set_title("Deg " + str(degree), fontsize=9, pad=3)
        ax.set_xlim(-1.05, 1.05)
        ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
        ax.tick_params(labelsize=7)
        
        # Create a single residual plot inset at bottom right of subplot
        # Position to avoid overlap with main dots, RMSE legend, and main figure legend
        # Size: 25% width, 20% height
        inset_width = 0.25  # 25% of subplot width
        inset_height = 0.20  # 20% of subplot height
        
        # Get axis limits and ranges
        x_lim = ax.get_xlim()
        y_lim = ax.get_ylim()
        x_range = x_lim[1] - x_lim[0]
        y_range = y_lim[1] - y_lim[0]
        x_min = x_lim[0]
        y_min = y_lim[0]
        
        # Position inset in right area: above RMSE legend and below main dots
        # RMSE text is at (0.95, 0.05) in axes coordinates (bottom right)
        # Place inset in right area, above RMSE legend
        
        # Convert main plot points to axes coordinates to find where dots are
        x_ax_coords = (x - x_min) / x_range
        classical_y_ax_coords = (classical - y_min) / y_range
        quantum_y_ax_coords = (quantum - y_min) / y_range
        
        # Find the bottom of main dots area (lowest y coordinate of main plot points)
        all_y_ax_coords = np.concatenate([classical_y_ax_coords, quantum_y_ax_coords])
        main_dots_bottom = np.min(all_y_ax_coords) if len(all_y_ax_coords) > 0 else 0.3
        
        # Position inset in right area
        # X: right side (around 0.65-0.70 to avoid RMSE text at 0.95)
        inset_x_pos = 0.68  # Right area
        
        # Y: above RMSE legend (0.05) but below main dots
        # RMSE legend extends to about 0.17 (0.05 + 0.12), so inset should be above that
        # But below main dots area
        rmse_top = 0.20  # Top of RMSE legend area
        inset_y_max = main_dots_bottom - 0.05  # Below main dots with buffer
        
        # Position inset in the gap between RMSE and main dots
        # If there's enough space, place it in the middle; otherwise near RMSE
        if inset_y_max > rmse_top + inset_height + 0.05:
            # Enough space: place in middle of gap
            inset_y_pos = (rmse_top + inset_y_max - inset_height) / 2
        else:
            # Limited space: place just above RMSE
            inset_y_pos = rmse_top + 0.02
        
        # Ensure inset doesn't go above main dots or below RMSE
        inset_y_pos = max(rmse_top + 0.02, min(inset_y_pos, main_dots_bottom - inset_height - 0.03))
        
        inset_ax = ax.inset_axes([inset_x_pos, inset_y_pos, inset_width, inset_height],
                                 transform=ax.transAxes)
        
        # Plot all residuals together
        n_points = len(x)
        x_indices = np.arange(n_points)
        
        # Use same colors as main plot (already defined above)
        
        # Plot classical residuals as dashed line (no markers)
        inset_ax.plot(x_indices, classical_residuals, 
                      color=classical_color, linestyle='--', linewidth=1.0,
                      alpha=0.8, zorder=2)
        
        # Plot quantum residuals sparsely (every 2nd or 3rd point) with error bars
        # Use error bars from q_err, but scale to show 0.001 scale more obviously
        sparse_step = max(2, n_points // 5)  # Show about 5 points, or every 2nd if fewer points
        sparse_indices = x_indices[::sparse_step]
        sparse_quantum_residuals = quantum_residuals[::sparse_step]
        sparse_q_err = q_err[::sparse_step]
        
        # Scale error bars to make 0.001 scale more visible
        # Ensure error bars are at least 0.001 in magnitude for visibility
        max_err = np.max(sparse_q_err) if len(sparse_q_err) > 0 else 0.001
        if max_err < 0.001:
            # Scale up small errors to be visible at 0.001 scale
            error_scale_factor = 0.001 / (max_err + 1e-10)
            error_scale_factor = min(error_scale_factor, 5.0)  # Cap at 5x to avoid distortion
        else:
            error_scale_factor = 1.0
        
        scaled_q_err = sparse_q_err * error_scale_factor
        
        # Plot quantum residuals with error bars, diamond shape with solid face (smaller)
        inset_ax.errorbar(sparse_indices, sparse_quantum_residuals, 
                         yerr=scaled_q_err,
                         fmt='D', color=quantum_color, 
                         markersize=2.5, capsize=1.5, capthick=0.6, elinewidth=0.6,
                         markeredgecolor=quantum_color, markeredgewidth=0.5,
                         markerfacecolor=quantum_color,  # Solid face
                         alpha=0.8, zorder=3)
        
        # Add zero reference line
        inset_ax.axhline(0, color='black', linestyle='-', linewidth=0.8, alpha=0.5, zorder=1)
        
        # Style the inset (remove xlabel, x ticks, x tick labels, and ylabel)
        inset_ax.set_xticks([])  # Remove x ticks
        inset_ax.tick_params(labelsize=4, pad=1)  # Smaller tick labels
        inset_ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.3)
        inset_ax.set_xlim(-0.5, n_points - 0.5)
        
        # Set y-limits based on residual range with some padding
        all_residuals = np.concatenate([classical_residuals, quantum_residuals])
        residual_range = np.max(np.abs(all_residuals)) if len(all_residuals) > 0 else 0.1
        inset_ax.set_ylim(-residual_range * 1.2, residual_range * 1.2)
        
        # Add subtle background
        inset_ax.set_facecolor('white')
        inset_ax.patch.set_alpha(0.9)
        for spine in inset_ax.spines.values():
            spine.set_linewidth(0.5)
            spine.set_color('0.5')
        
        # if idx == 0:
        #     ax.legend(loc='best', fontsize=8, framealpha=0.95)
        
        # Add RMSE annotation (concise)
        rmse_text = "RMSE: " + "{:.3f}".format(data["quantum_rmse_mean"])
        ax.text(0.95, 0.05, rmse_text, transform=ax.transAxes,
               fontsize=7, ha='right', va='bottom',
               bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8, edgecolor='0.6', linewidth=0.5))
    
    # Global legend at the top
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, 0.98), 
              ncol=3, fontsize=8, frameon=False, handletextpad=0.4, columnspacing=1.0)

    plt.tight_layout(pad=0.4, h_pad=0.4, w_pad=0.4, rect=[0, 0, 1, 0.98])
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight', pad_inches=0.02)
    pdf_path = output_path.rsplit('.', 1)[0] + '.pdf'
    plt.savefig(pdf_path, dpi=300, bbox_inches='tight', pad_inches=0.02)
    
    print("Figure 2 saved to: " + output_path)
    plt.close()


# ==============================================================================
# FIGURE 3: ERROR DISTRIBUTION (3x2 for single column)
# ==============================================================================

def create_figure3_error_distribution(aggregated, output_path):
    """Create Figure 3: Error distribution histograms by degree (single-column)."""
    configure_single_column_style()
    
    colors = PLOT_CONFIG['colors']
    degrees = sorted(aggregated.keys())
    
    fig, axes = plt.subplots(3, 2, figsize=(SINGLE_COL_WIDTH, 6.5))
    axes = axes.flatten()
    
    for idx, degree in enumerate(degrees):
        ax = axes[idx]
        data = aggregated[degree]
        
        all_quantum_errors = []
        all_classical_errors = []
        
        for trial_data in data['results']:
            q_errors = trial_data['measured'] - trial_data['theoretical']
            c_errors = trial_data['classical_pred'] - trial_data['theoretical']
            all_quantum_errors.extend(q_errors)
            all_classical_errors.extend(c_errors)
        
        all_quantum_errors = np.array(all_quantum_errors)
        all_classical_errors = np.array(all_classical_errors)
        
        bins = np.linspace(-0.1, 0.1, 25)
        
        ax.hist(all_classical_errors, bins=bins, color=colors['classical'],
               alpha=0.5, label='Classical', edgecolor='black', linewidth=0.3, density=True)
        ax.hist(all_quantum_errors, bins=bins, color=colors['quantum'],
               alpha=0.5, label='Quantum', edgecolor='black', linewidth=0.3, density=True)
        
        ax.axvspan(-0.03, 0.03, color=colors.get('pass_region', 'green'), alpha=0.2)
        ax.axvline(0, color='black', linestyle='-', linewidth=0.8)
        
        ax.set_xlabel('Error', fontsize=10)
        ax.set_ylabel('Density', fontsize=10)
        ax.set_title("Deg " + str(degree), fontsize=10, pad=5)
        ax.set_xlim(-0.1, 0.1)
        ax.tick_params(labelsize=8)
        
        # Add pass rate (concise) - position to avoid legend overlap
        pass_text = "{:.0f}%".format(data["pass_rate_mean"]*100)
        # Position pass rate text in upper left to avoid legend in upper right
        ax.text(0.05, 0.95, pass_text, transform=ax.transAxes,
               fontsize=8, ha='left', va='top',
               bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8, edgecolor='0.6', linewidth=0.5))
    
    # Place legend outside subplots to avoid overlap
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.99), 
              ncol=2, fontsize=8, framealpha=0.95, frameon=True)
    
    plt.tight_layout(pad=0.8, h_pad=0.6, w_pad=0.6, rect=[0, 0, 1, 0.96])
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight', pad_inches=0.02)
    pdf_path = output_path.rsplit('.', 1)[0] + '.pdf'
    plt.savefig(pdf_path, dpi=300, bbox_inches='tight', pad_inches=0.02)
    
    print("Figure 3 saved to: " + output_path)
    plt.close()


# ==============================================================================
# FIGURE 4: CORRELATION SUMMARY (Stacked vertically)
# ==============================================================================

def create_figure4_correlation_summary(aggregated, all_data, output_path):
    """Create Figure 4: Correlation analysis (stacked, single-column)."""
    configure_single_column_style()
    
    colors = PLOT_CONFIG['colors']
    degrees = sorted(aggregated.keys())
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(SINGLE_COL_WIDTH, 5))
    
    # Panel A: Scatter plot
    degree_colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(degrees)))
    
    for idx, degree in enumerate(degrees):
        data = aggregated[degree]
        
        for trial_data in data['results']:
            theo = trial_data['theoretical']
            quantum = trial_data['measured']
            
            ax1.scatter(theo, quantum, c=[degree_colors[idx]], s=15, alpha=0.5,
                       label="Deg " + str(degree) if trial_data == data['results'][0] else '',
                       edgecolors='black', linewidth=0.2)
    
    ax1.plot([-0.6, 0.6], [-0.6, 0.6], 'k--', linewidth=1.2, label='Ideal', alpha=0.7)
    
    ax1.set_xlabel('Theoretical', fontsize=10)
    ax1.set_ylabel('Quantum Measured', fontsize=10)
    ax1.set_title('(a) Measured vs Theory', fontsize=10, pad=8)
    ax1.set_aspect('equal')
    ax1.set_xlim(-0.6, 0.6)
    ax1.set_ylim(-0.6, 0.6)
    # Move legend to upper left to avoid overlap with data points
    ax1.legend(loc='upper left', fontsize=7, ncol=4, framealpha=0.95, 
              columnspacing=0.8, handletextpad=0.3)
    ax1.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    ax1.tick_params(labelsize=9)
    
    # Panel B: Metrics by degree
    pass_rates = [aggregated[d]['pass_rate_mean'] * 100 for d in degrees]
    q_corr = [aggregated[d]['quantum_corr_mean'] for d in degrees]
    
    ax2_twin = ax2.twinx()
    
    bars = ax2.bar(degrees, pass_rates, color=colors['quantum'], alpha=0.7,
                  edgecolor='black', linewidth=0.4, label='Pass Rate', width=0.6)
    ax2.set_ylabel('Pass Rate (%)', color=colors['quantum'], fontsize=10)
    ax2.set_ylim(0, 110)
    ax2.tick_params(axis='y', labelcolor=colors['quantum'], labelsize=9)
    
    ax2_twin.plot(degrees, q_corr, 'o-', color=colors['theoretical'],
                 linewidth=1.5, markersize=5, label='Correlation', markeredgecolor='black', markeredgewidth=0.4)
    ax2_twin.set_ylabel('Correlation', color=colors['theoretical'], fontsize=10)
    ax2_twin.set_ylim(0.99, 1.001)
    ax2_twin.tick_params(axis='y', labelcolor=colors['theoretical'], labelsize=9)
    
    ax2.set_xlabel('Degree', fontsize=10)
    ax2.set_title('(b) Performance Metrics', fontsize=10, pad=8)
    ax2.set_xticks(degrees)
    ax2.tick_params(labelsize=9)
    
    # Combined legend - move to upper right to avoid overlap with bars
    lines1, labels1 = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2_twin.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize=8, 
              framealpha=0.95, frameon=True)
    
    ax2.grid(True, alpha=0.3, axis='y', linestyle='--', linewidth=0.5)
    
    plt.tight_layout(pad=0.8, h_pad=1.2)
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight', pad_inches=0.02)
    pdf_path = output_path.rsplit('.', 1)[0] + '.pdf'
    plt.savefig(pdf_path, dpi=300, bbox_inches='tight', pad_inches=0.02)
    
    print("Figure 4 saved to: " + output_path)
    plt.close()


# ==============================================================================
# LATEX TABLE
# ==============================================================================

def generate_latex_table(aggregated, output_path):
    """Generate LaTeX table of results (RevTeX compatible)."""
    degrees = sorted(aggregated.keys())
    
    latex = r"""
\begin{table}[ht]
\centering
\caption{Polynomial Recovery Accuracy: Native Quantum Arithmetic}
\label{tab:results-native}
\begin{tabular}{lccccc}
\toprule
Degree & Name & Q-RMSE & C-RMSE & Correlation & Pass Rate \\
\midrule
"""
    
    for degree in degrees:
        data = aggregated[degree]
        q_rmse = "${:.4f} \\pm {:.4f}$".format(data['quantum_rmse_mean'], data['quantum_rmse_std'])
        c_rmse = "${:.4f} \\pm {:.4f}$".format(data['classical_rmse_mean'], data['classical_rmse_std'])
        corr = "${:.4f}$".format(data['quantum_corr_mean'])
        pass_rate = "${:.1f}\\%$".format(data['pass_rate_mean']*100)
        
        latex += str(degree) + " & " + data['name'] + " & " + q_rmse + " & " + c_rmse + " & " + corr + " & " + pass_rate + " \\\\\n"
    
    latex += r"""\bottomrule
\end{tabular}
\end{table}
"""
    
    with open(output_path, 'w') as f:
        f.write(latex)
    
    print("LaTeX table saved to: " + output_path)
    return latex


# ==============================================================================
# MAIN
# ==============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description='Generate single-column publication figures for RevTeX'
    )
    parser.add_argument('--input', default='results/cloud',
                       help='Input directory with H5 result files')
    parser.add_argument('--output', default='figures/paper/single_column',
                       help='Output directory for figures')
    parser.add_argument('--approach', default='native', choices=['native', 'direct', 'all'],
                       help='Which approach to load: native, direct, or all (default: native)')
    parser.add_argument('--ionq-input', default=None,
                       help='Input directory for IonQ stress test data (degrees 1-35). If specified, fig1 will use IonQ data. Default: research/ionq/results/stress_test')
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
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Project root is one level up from research/
    project_root = os.path.dirname(script_dir)
    
    # Resolve input path (handle both absolute and relative)
    if os.path.isabs(args.input):
        input_dir = args.input
    elif os.path.exists(args.input):
        # Path exists as-is, use it
        input_dir = os.path.abspath(args.input)
    elif os.path.exists(os.path.join(script_dir, args.input)):
        # Path exists relative to script dir
        input_dir = os.path.join(script_dir, args.input)
    elif os.path.exists(os.path.join(project_root, args.input)):
        # Path exists relative to project root
        input_dir = os.path.join(project_root, args.input)
    else:
        # Default: relative to script dir
        input_dir = os.path.join(script_dir, args.input)
    
    # Resolve output path (default to script_dir-relative)
    if os.path.isabs(args.output):
        output_dir = args.output
    elif os.path.exists(args.output):
        # Path exists as-is, use it
        output_dir = os.path.abspath(args.output)
    elif os.path.exists(os.path.join(script_dir, args.output)):
        # Path exists relative to script dir
        output_dir = os.path.join(script_dir, args.output)
    elif os.path.exists(os.path.join(project_root, args.output)):
        # Path exists relative to project root
        output_dir = os.path.join(project_root, args.output)
    else:
        # Default: relative to script dir
        output_dir = os.path.join(script_dir, args.output)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Load data
    filter_approach = None if args.approach == 'all' else args.approach
    all_data = load_native_results(input_dir, filter_approach=filter_approach)
    if not all_data:
        print("No " + args.approach + " approach data found in " + input_dir + "!")
        print("Available files: " + str(glob.glob(os.path.join(input_dir, 'deg*.h5'))[:5]))
        sys.exit(1)
    
    aggregated = aggregate_by_degree(all_data)
    
    # Load IonQ data for fig1 if specified or if fig1 is requested
    ionq_aggregated = None
    if args.ionq_input or args.fig1 or args.all:
        # Use provided path or default to research/ionq/results/stress_test
        ionq_input_dir = args.ionq_input if args.ionq_input else 'research/ionq/results/stress_test'
        
        if not os.path.isabs(ionq_input_dir):
            # Try relative to script dir or project root
            if os.path.exists(os.path.join(script_dir, ionq_input_dir)):
                ionq_input_dir = os.path.join(script_dir, ionq_input_dir)
            elif os.path.exists(os.path.join(project_root, ionq_input_dir)):
                ionq_input_dir = os.path.join(project_root, ionq_input_dir)
            else:
                # Try relative to research/ionq directory
                ionq_dir = os.path.join(project_root, 'research', 'ionq')
                potential_path = os.path.join(ionq_dir, 'results', 'stress_test')
                if os.path.exists(potential_path):
                    ionq_input_dir = potential_path
        
        ionq_all_data = load_ionq_stress_test_results(ionq_input_dir)
        if ionq_all_data:
            ionq_aggregated = aggregate_by_degree(ionq_all_data)
            print(f"Loaded IonQ stress test data for fig1 (degrees 1-35) from {ionq_input_dir}")
        else:
            if args.ionq_input:
                print(f"Warning: No IonQ stress test data found at {ionq_input_dir}, using regular data for fig1")
            else:
                print(f"Note: No IonQ stress test data found at {ionq_input_dir}, using regular data for fig1")
                print(f"  (To use IonQ data, specify --ionq-input or ensure data is in {ionq_input_dir})")
    
    generate_all = args.all or not (args.fig1 or args.fig2 or args.fig3 or args.fig4 or args.tables)
    
    print("\n" + "=" * 60)
    print("GENERATING SINGLE-COLUMN REVTEX FIGURES")
    print("=" * 60)
    
    if generate_all or args.fig1:
        fig1_path = os.path.join(output_dir, 'fig1_degree_scaling.png')
        # Use IonQ data if available, otherwise use regular data
        fig1_aggregated = ionq_aggregated if ionq_aggregated is not None else aggregated
        create_figure1_degree_scaling(fig1_aggregated, fig1_path)
    
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
    print("All outputs saved to: " + output_dir)
    print("=" * 60)
    
    if not args.no_display:
        plt.show()
