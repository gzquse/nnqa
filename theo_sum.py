#!/usr/bin/env python3
"""
Neural-Native Quantum Arithmetic: Theoretical Bounds
For ICML Submission - Updated Figure Generation
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# ==============================================================================
# THEORETICAL ERROR CALCULATIONS
# ==============================================================================

def shot_noise_std(n_shots: int, p: float = 0.5) -> float:
    return 2 * np.sqrt(p * (1 - p) / n_shots)

# ==============================================================================
# FIGURE 1: SHOT NOISE ANALYSIS (FINAL - 1x2 Layout)
# ==============================================================================

def create_shot_noise_figure(save_path: str = None):
    """
    Create figure showing shot noise as the fundamental error source.
    Formatted for ICML (10pt body text) with compact dimensions.
    Layout: 1 Row, 2 Columns.
    """
    # Configure for compact column width
    # Width increased to 3.25 (standard ICML column) to fit two plots side-by-side
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'DejaVu Serif'],
        'font.size': 6,
        'axes.labelsize': 6,
        'axes.titlesize': 6,
        'legend.fontsize': 5,
        'xtick.labelsize': 5,
        'ytick.labelsize': 5,
        'lines.linewidth': 1.0,
        'axes.linewidth': 0.8,
        'figure.dpi': 300,
        # Global spine settings
        'axes.spines.top': False,
        'axes.spines.right': False,
    })
    
    # CHANGED: 1 row, 2 columns. 
    # Width 3.25 (full single column width), Height 1.5 (short)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(3.25, 1.5))
    
    colors = {
        'theory': '#2E8B57',
        'worst': '#DC267F', 
        'typical': '#648FFF',
    }
    
    # =========================================================================
    # Panel (a): Error vs Shots
    # =========================================================================
    
    shots_range = np.logspace(2, 5, 100)
    sigma_worst = [shot_noise_std(int(n), 0.5) for n in shots_range]
    sigma_typical = [shot_noise_std(int(n), 0.35) for n in shots_range]
    sigma_best = [shot_noise_std(int(n), 0.1) for n in shots_range]
    
    # Plot curves
    ax1.loglog(shots_range, sigma_worst, '-', color=colors['worst'], 
              label=r'$p=0.5$')
    ax1.loglog(shots_range, sigma_typical, '-', color=colors['typical'],
              label=r'$p=0.35$')
    ax1.loglog(shots_range, sigma_best, '-', color=colors['theory'],
              label=r'$p=0.1$')
    
    # Annotate 4K point
    n_mark = 4096
    sigma_mark = shot_noise_std(n_mark, 0.35)
    ax1.scatter([n_mark], [sigma_mark], s=7, color=colors['typical'], 
                zorder=5, edgecolors='k', linewidth=0.5)
    ax1.annotate('4K', (n_mark, sigma_mark), xytext=(4, 2), 
                 textcoords='offset points', fontsize=5, ha='center')

    ax1.set_xlabel('Shots $N$', labelpad=1)
    ax1.set_ylabel(r'Std. Dev. $\sigma$', labelpad=1)
    ax1.set_title('(a) Shot Noise vs $N$', pad=3)
    
    # Legend: Upper right, no frame
    ax1.legend(loc='upper right', frameon=False, handlelength=1.0, 
               handletextpad=0.3, borderaxespad=0.1)
    
    ax1.set_xlim(100, 100000)
    ax1.set_ylim(0.002, 0.3)
    ax1.grid(True, which='major', alpha=0.3)
    
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)

    # =========================================================================
    # Panel (b): Error Distribution
    # =========================================================================
    
    n_shots = 4096
    true_p = 0.35
    
    # Theoretical Gaussian
    sigma_theory = shot_noise_std(n_shots, true_p)
    x_gauss = np.linspace(-0.06, 0.06, 200)
    y_gauss = (1 / (sigma_theory * np.sqrt(2 * np.pi))) * \
              np.exp(-x_gauss**2 / (2 * sigma_theory**2))
    
    # Simulate data
    np.random.seed(42)
    measured_counts = np.random.binomial(n_shots, true_p, 10000)
    measured_z = 1 - 2 * (measured_counts / n_shots)
    true_z = 1 - 2 * true_p
    errors = measured_z - true_z

    ax2.hist(errors, bins=30, density=True, color=colors['typical'],
            alpha=0.6, edgecolor='none')
    ax2.plot(x_gauss, y_gauss, '-', color=colors['theory'], linewidth=1.5,
            label='Theory')
    
    ax2.set_xlabel('Error', labelpad=1)
    ax2.set_ylabel('Density', labelpad=1)
    ax2.set_title(f'(b) Error Dist. (4K)', pad=3)
    
    # Legend: No frame
    ax2.legend(loc='upper left', frameon=False, handlelength=1.0, borderaxespad=0)
    
    ax2.set_xlim(-0.07, 0.07)
    ax2.set_xticks([-0.06, 0, 0.06])
    
    # Stats text
    emp_std = np.std(errors)
    ax2.text(0.98, 0.95, f'$\\sigma={emp_std:.3f}$',
            transform=ax2.transAxes, fontsize=5, va='top', ha='right')
    
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    
    # Tight layout with w_pad for side-by-side spacing
    plt.tight_layout(pad=0.2, w_pad=1.0)
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        print(f"Figure saved: {save_path}")
    
    return fig

# ==============================================================================
# MAIN
# ==============================================================================

if __name__ == "__main__":
    import os
    os.makedirs('out', exist_ok=True)
    
    print("Generating ICML formatted figure (1 Row x 2 Columns)...")
    create_shot_noise_figure('out/icml_shot_noise_1x2.pdf')
    create_shot_noise_figure('out/icml_shot_noise_1x2.png')