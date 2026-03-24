#!/usr/bin/env python3
"""
Polynomial Recovery Metrics with Error Bars
For Neural-Native Quantum Arithmetic Demo

Based on recovery method from ehands_2q.py:
- Compares theoretical expected values (tEV) with measured expected values (mEV)
- tEV is computed from the polynomial formula
- mEV = 1 - 2*mprob where mprob = n1/(n0+n1)

Plotting style follows theo_sum.py template for publication quality.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import argparse
import os
import sys

# Add parent path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit_aer import AerSimulator


# ==============================================================================
# QUANTUM ARITHMETIC PRIMITIVES (from nn_to_quantum.py)
# ==============================================================================

def data_to_angle(x):
    """Convert x in [-1,1] to rotation angle. After Ry(theta), <Z> = cos(theta) = x."""
    x = np.clip(x, -1 + 1e-7, 1 - 1e-7)
    return np.arccos(x)


def weight_to_alpha(w):
    """Convert weight w in [0,1] to alpha angle for weighted sum."""
    w = np.clip(w, 1e-7, 1 - 1e-7)
    return np.arccos(1 - 2*w)


def run_circuit_with_counts(qc, shots=8192):
    """Execute circuit and return raw counts."""
    backend = AerSimulator()
    qc_t = transpile(qc, backend, optimization_level=1)
    job = backend.run(qc_t, shots=shots)
    return job.result().get_counts()


def quantum_polynomial_direct(x, coefficients, shots=8192):
    """
    Direct quantum polynomial evaluation with raw counts for error analysis.
    
    Returns: (mEV, counts_dict)
    """
    # Classical evaluation
    y = sum(coefficients[i] * (x ** i) for i in range(len(coefficients)))
    y_clipped = np.clip(y, -1 + 1e-6, 1 - 1e-6)
    
    # Encode and measure
    qr = QuantumRegister(1)
    cr = ClassicalRegister(1)
    qc = QuantumCircuit(qr, cr)
    
    qc.ry(data_to_angle(y_clipped), 0)
    qc.measure(0, 0)
    
    counts = run_circuit_with_counts(qc, shots)
    n0, n1 = counts.get('0', 0), counts.get('1', 0)
    mEV = (n0 - n1) / shots
    
    return mEV, counts


def quantum_multiplication(x0, x1, shots=8192):
    """Quantum multiplication with raw counts."""
    qr = QuantumRegister(2)
    cr = ClassicalRegister(1)
    qc = QuantumCircuit(qr, cr)
    
    qc.ry(data_to_angle(x0), 0)
    qc.ry(data_to_angle(x1), 1)
    qc.barrier()
    
    qc.rz(np.pi/2, 1)
    qc.cx(0, 1)
    qc.measure(1, 0)
    
    counts = run_circuit_with_counts(qc, shots)
    n0, n1 = counts.get('0', 0), counts.get('1', 0)
    mEV = (n0 - n1) / shots
    
    return mEV, counts


def quantum_weighted_sum(x0, x1, w, shots=8192):
    """Quantum weighted sum with raw counts."""
    qr = QuantumRegister(2)
    cr = ClassicalRegister(1)
    qc = QuantumCircuit(qr, cr)
    
    qc.ry(data_to_angle(x0), 0)
    qc.ry(data_to_angle(x1), 1)
    qc.barrier()
    
    alpha = weight_to_alpha(w)
    qc.rz(np.pi/2, 1)
    qc.cx(0, 1)
    qc.ry(alpha/2, 0)
    qc.cx(1, 0)
    qc.ry(-alpha/2, 0)
    
    qc.measure(0, 0)
    
    counts = run_circuit_with_counts(qc, shots)
    n0, n1 = counts.get('0', 0), counts.get('1', 0)
    mEV = (n0 - n1) / shots
    
    return mEV, counts


# ==============================================================================
# RECOVERY ERROR CALCULATION (from ehands_2q.py methodology)
# ==============================================================================

def compute_recovery_error(counts, theoretical_value):
    """
    Compute recovery error following ehands_2q.py methodology.
    
    mEV = 1 - 2*mprob where mprob = n1/(n0+n1)
    diff = tEV - mEV
    
    Returns: (mEV, error, std_error)
    """
    n0, n1 = counts.get('0', 0), counts.get('1', 0)
    shots = n0 + n1
    mprob = n1 / shots
    
    # Measured expectation value
    mEV = 1 - 2 * mprob
    
    # Recovery error
    error = theoretical_value - mEV
    
    # Standard error from binomial distribution
    # Var(mprob) = p*(1-p)/N, so Var(mEV) = 4*Var(mprob)
    std_error = 2 * np.sqrt(mprob * (1 - mprob) / shots)
    
    return mEV, error, std_error


def run_recovery_trials(x, coefficients, n_trials=10, shots=2048):
    """
    Run multiple recovery trials to get statistics.
    
    Returns: (mean_mEV, mean_error, std_error, all_errors)
    """
    tEV = sum(coefficients[i] * (x ** i) for i in range(len(coefficients)))
    tEV_clipped = np.clip(tEV, -1 + 1e-6, 1 - 1e-6)
    
    all_mEV = []
    all_errors = []
    
    for _ in range(n_trials):
        mEV, counts = quantum_polynomial_direct(x, coefficients, shots)
        error = tEV_clipped - mEV
        all_mEV.append(mEV)
        all_errors.append(error)
    
    mean_mEV = np.mean(all_mEV)
    std_mEV = np.std(all_mEV)
    mean_error = np.mean(all_errors)
    std_error = np.std(all_errors)
    
    return mean_mEV, std_mEV, mean_error, std_error, all_errors, tEV_clipped


# ==============================================================================
# MATPLOTLIB CONFIGURATION (from theo_sum.py)
# ==============================================================================

def configure_matplotlib_for_publication():
    """Configure matplotlib for publication-quality figures."""
    plt.rcParams.update({
        # Font settings
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'DejaVu Serif', 'Computer Modern Roman'],
        'font.size': 11,
        'axes.labelsize': 12,
        'axes.titlesize': 13,
        'legend.fontsize': 10,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        
        # Figure aesthetics
        'figure.figsize': (7, 5),
        'figure.dpi': 150,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.05,
        
        # Axes
        'axes.linewidth': 0.8,
        'axes.grid': True,
        'grid.alpha': 0.3,
        'grid.linewidth': 0.5,
        
        # Lines
        'lines.linewidth': 2.0,
        'lines.markersize': 6,
        
        # Legend
        'legend.framealpha': 0.95,
        'legend.edgecolor': '0.8',
        'legend.fancybox': False,
    })


# ==============================================================================
# MAIN PLOTTING FUNCTIONS
# ==============================================================================

def create_recovery_plot(coefficients, n_points=15, n_trials=10, shots=2048, 
                        save_path=None):
    """
    Create publication-quality polynomial recovery plot with error bars.
    
    Parameters:
        coefficients: Polynomial coefficients [a0, a1, a2, ...]
        n_points: Number of test points
        n_trials: Number of trials per point for statistics
        shots: Shots per circuit execution
        save_path: Path to save figure
    """
    configure_matplotlib_for_publication()
    
    # Colorblind-friendly palette (IBM Design)
    colors = {
        'theoretical': '#2E8B57',      # Sea Green
        'measured': '#DC267F',          # Magenta
        'error_fill': '#785EF0',        # Purple
        'zero_line': '#808080',         # Gray
        'pass_region': '#90EE90',       # Light Green
        'poor_region': '#FFE4B5',       # Moccasin
        'fail_region': '#FFB6C1',       # Light Pink
    }
    
    # Generate test points
    x_test = np.linspace(-0.9, 0.9, n_points)
    x_continuous = np.linspace(-1, 1, 200)
    
    # Compute theoretical curve
    y_theoretical = np.array([
        np.clip(sum(coefficients[i] * (x ** i) for i in range(len(coefficients))), 
                -1 + 1e-6, 1 - 1e-6)
        for x in x_continuous
    ])
    
    # Run recovery trials
    print("Running recovery trials...")
    results = {
        'x': [],
        'tEV': [],
        'mean_mEV': [],
        'std_mEV': [],
        'mean_error': [],
        'std_error': [],
    }
    
    for i, x in enumerate(x_test):
        mean_mEV, std_mEV, mean_error, std_error, _, tEV = run_recovery_trials(
            x, coefficients, n_trials, shots
        )
        results['x'].append(x)
        results['tEV'].append(tEV)
        results['mean_mEV'].append(mean_mEV)
        results['std_mEV'].append(std_mEV)
        results['mean_error'].append(mean_error)
        results['std_error'].append(std_error)
        
        print(f"  x={x:6.2f}: tEV={tEV:7.4f}, mEV={mean_mEV:7.4f} +/- {std_mEV:.4f}, "
              f"error={mean_error:+7.4f} +/- {std_error:.4f}")
    
    # Convert to numpy arrays
    for key in results:
        results[key] = np.array(results[key])
    
    # ==========================================================================
    # CREATE FIGURE WITH SUBPLOTS
    # ==========================================================================
    
    fig, (ax_poly, ax_error) = plt.subplots(1, 2, figsize=(12, 5))
    
    # ==========================================================================
    # LEFT PANEL: Polynomial Recovery
    # ==========================================================================
    
    # Plot theoretical curve
    ax_poly.plot(x_continuous, y_theoretical, 
                color=colors['theoretical'], linewidth=2.5, 
                label='Theoretical $F(x)$', zorder=3)
    
    # Plot measured values with error bars
    ax_poly.errorbar(results['x'], results['mean_mEV'], 
                    yerr=results['std_mEV'],
                    fmt='o', color=colors['measured'],
                    markersize=7, capsize=4, capthick=1.5,
                    elinewidth=1.5, markeredgecolor='black',
                    markeredgewidth=0.5, label='Quantum Measured',
                    zorder=4)
    
    # Fill between for confidence region
    ax_poly.fill_between(results['x'], 
                        results['mean_mEV'] - results['std_mEV'],
                        results['mean_mEV'] + results['std_mEV'],
                        color=colors['error_fill'], alpha=0.2,
                        label=r'$\pm 1\sigma$ region')
    
    # Formatting
    ax_poly.set_xlabel(r'Input $x$')
    ax_poly.set_ylabel(r'Expected Value $\langle Z \rangle$')
    ax_poly.set_title('(a) Polynomial Recovery: Theoretical vs Measured')
    ax_poly.set_xlim(-1, 1)
    ax_poly.legend(loc='best', fontsize=9)
    ax_poly.grid(True, alpha=0.3)
    
    # Add polynomial formula
    poly_str = _format_polynomial(coefficients)
    ax_poly.text(0.03, 0.97, f'$F(x) = {poly_str}$', 
                transform=ax_poly.transAxes,
                fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='wheat', alpha=0.7))
    
    # ==========================================================================
    # RIGHT PANEL: Recovery Error Analysis
    # ==========================================================================
    
    # Add threshold regions (from ehands_2q.py: PASS < 0.03, POOR < 0.1, FAIL >= 0.1)
    ax_error.axhspan(-0.03, 0.03, color=colors['pass_region'], alpha=0.3, 
                    label='PASS region ($|\\delta| < 0.03$)')
    ax_error.axhspan(-0.1, -0.03, color=colors['poor_region'], alpha=0.3)
    ax_error.axhspan(0.03, 0.1, color=colors['poor_region'], alpha=0.3,
                    label='POOR region ($|\\delta| < 0.10$)')
    ax_error.axhspan(-0.3, -0.1, color=colors['fail_region'], alpha=0.3)
    ax_error.axhspan(0.1, 0.3, color=colors['fail_region'], alpha=0.3,
                    label='FAIL region ($|\\delta| \\geq 0.10$)')
    
    # Zero line
    ax_error.axhline(y=0, color=colors['zero_line'], linestyle='-', 
                    linewidth=1.5, alpha=0.7)
    
    # Plot recovery errors with error bars
    ax_error.errorbar(results['x'], results['mean_error'],
                     yerr=results['std_error'],
                     fmt='s', color=colors['measured'],
                     markersize=7, capsize=4, capthick=1.5,
                     elinewidth=1.5, markeredgecolor='black',
                     markeredgewidth=0.5, 
                     label=r'Recovery Error $\delta = t_{EV} - m_{EV}$',
                     zorder=4)
    
    # Connect points with line
    ax_error.plot(results['x'], results['mean_error'], 
                 color=colors['measured'], linewidth=1.0, alpha=0.5)
    
    # Formatting
    ax_error.set_xlabel(r'Input $x$')
    ax_error.set_ylabel(r'Recovery Error $\delta$')
    ax_error.set_title('(b) Recovery Error Analysis')
    ax_error.set_xlim(-1, 1)
    ax_error.set_ylim(-0.15, 0.15)
    ax_error.legend(loc='upper right', fontsize=8)
    ax_error.grid(True, alpha=0.3)
    
    # Add statistics box
    mean_abs_error = np.mean(np.abs(results['mean_error']))
    max_abs_error = np.max(np.abs(results['mean_error']))
    rmse = np.sqrt(np.mean(results['mean_error']**2))
    
    stats_text = (f"Mean $|\\delta|$: {mean_abs_error:.4f}\n"
                 f"Max $|\\delta|$: {max_abs_error:.4f}\n"
                 f"RMSE: {rmse:.4f}\n"
                 f"Shots: {shots}, Trials: {n_trials}")
    ax_error.text(0.03, 0.03, stats_text,
                 transform=ax_error.transAxes,
                 fontsize=9, verticalalignment='bottom',
                 bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.9))
    
    # ==========================================================================
    # FINAL ADJUSTMENTS
    # ==========================================================================
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to: {save_path}")
    
    plt.show()
    
    return fig, results


def create_operation_comparison_plot(n_trials=20, shots=4096, save_path=None):
    """
    Create comparison plot for quantum arithmetic operations:
    - Weighted Sum
    - Multiplication
    
    Shows recovery accuracy for each operation type.
    """
    configure_matplotlib_for_publication()
    
    colors = {
        'weighted_sum': '#648FFF',   # Blue
        'multiplication': '#DC267F', # Magenta
        'zero_line': '#808080',
        'pass_region': '#90EE90',
    }
    
    # Test cases for weighted sum: y = w*x0 + (1-w)*x1
    ws_test_cases = [
        (0.8, -0.3, 0.2),   # x0, x1, w
        (0.5, 0.5, 0.5),
        (-0.6, 0.4, 0.7),
        (0.0, 0.9, 0.1),
        (-0.9, -0.9, 0.5),
    ]
    
    # Test cases for multiplication: y = x0 * x1
    mult_test_cases = [
        (0.8, 0.5),
        (-0.6, 0.7),
        (0.3, -0.4),
        (-0.5, -0.5),
        (0.9, 0.1),
    ]
    
    fig, (ax_ws, ax_mult) = plt.subplots(1, 2, figsize=(12, 5))
    
    # ==========================================================================
    # LEFT: Weighted Sum Recovery
    # ==========================================================================
    print("Testing Weighted Sum operations...")
    ws_results = {'x_idx': [], 'tEV': [], 'mean_mEV': [], 'std_mEV': [], 'errors': []}
    
    for idx, (x0, x1, w) in enumerate(ws_test_cases):
        tEV = w * x0 + (1 - w) * x1
        all_mEV = []
        
        for _ in range(n_trials):
            mEV, _ = quantum_weighted_sum(x0, x1, w, shots)
            all_mEV.append(mEV)
        
        mean_mEV = np.mean(all_mEV)
        std_mEV = np.std(all_mEV)
        error = tEV - mean_mEV
        
        ws_results['x_idx'].append(idx)
        ws_results['tEV'].append(tEV)
        ws_results['mean_mEV'].append(mean_mEV)
        ws_results['std_mEV'].append(std_mEV)
        ws_results['errors'].append(error)
        
        print(f"  WS[{idx}]: ({x0:.1f},{x1:.1f},{w:.1f}) -> tEV={tEV:.4f}, "
              f"mEV={mean_mEV:.4f}+/-{std_mEV:.4f}")
    
    # Plot weighted sum results
    ax_ws.axhspan(-0.03, 0.03, color=colors['pass_region'], alpha=0.3)
    ax_ws.axhline(y=0, color=colors['zero_line'], linestyle='-', linewidth=1)
    
    ax_ws.errorbar(ws_results['x_idx'], ws_results['tEV'],
                   fmt='o', color='black', markersize=8,
                   label='Theoretical', zorder=5)
    ax_ws.errorbar(ws_results['x_idx'], ws_results['mean_mEV'],
                   yerr=ws_results['std_mEV'],
                   fmt='s', color=colors['weighted_sum'],
                   markersize=8, capsize=5, capthick=2,
                   elinewidth=2, markeredgecolor='black',
                   markeredgewidth=0.5, label='Measured',
                   zorder=4)
    
    # Add case labels
    case_labels = [f"({x0},{x1},{w})" for x0, x1, w in ws_test_cases]
    ax_ws.set_xticks(ws_results['x_idx'])
    ax_ws.set_xticklabels(case_labels, rotation=45, ha='right', fontsize=8)
    
    ax_ws.set_xlabel(r'Test Case $(x_0, x_1, w)$')
    ax_ws.set_ylabel(r'Expected Value $\langle Z \rangle$')
    ax_ws.set_title(r'(a) Weighted Sum: $\langle Z \rangle = w \cdot x_0 + (1-w) \cdot x_1$')
    ax_ws.legend(loc='best')
    ax_ws.grid(True, alpha=0.3)
    
    # ==========================================================================
    # RIGHT: Multiplication Recovery
    # ==========================================================================
    print("\nTesting Multiplication operations...")
    mult_results = {'x_idx': [], 'tEV': [], 'mean_mEV': [], 'std_mEV': [], 'errors': []}
    
    for idx, (x0, x1) in enumerate(mult_test_cases):
        tEV = x0 * x1
        all_mEV = []
        
        for _ in range(n_trials):
            mEV, _ = quantum_multiplication(x0, x1, shots)
            all_mEV.append(mEV)
        
        mean_mEV = np.mean(all_mEV)
        std_mEV = np.std(all_mEV)
        error = tEV - mean_mEV
        
        mult_results['x_idx'].append(idx)
        mult_results['tEV'].append(tEV)
        mult_results['mean_mEV'].append(mean_mEV)
        mult_results['std_mEV'].append(std_mEV)
        mult_results['errors'].append(error)
        
        print(f"  MULT[{idx}]: ({x0:.1f},{x1:.1f}) -> tEV={tEV:.4f}, "
              f"mEV={mean_mEV:.4f}+/-{std_mEV:.4f}")
    
    # Plot multiplication results
    ax_mult.axhspan(-0.03, 0.03, color=colors['pass_region'], alpha=0.3)
    ax_mult.axhline(y=0, color=colors['zero_line'], linestyle='-', linewidth=1)
    
    ax_mult.errorbar(mult_results['x_idx'], mult_results['tEV'],
                     fmt='o', color='black', markersize=8,
                     label='Theoretical', zorder=5)
    ax_mult.errorbar(mult_results['x_idx'], mult_results['mean_mEV'],
                     yerr=mult_results['std_mEV'],
                     fmt='s', color=colors['multiplication'],
                     markersize=8, capsize=5, capthick=2,
                     elinewidth=2, markeredgecolor='black',
                     markeredgewidth=0.5, label='Measured',
                     zorder=4)
    
    # Add case labels
    case_labels = [f"({x0},{x1})" for x0, x1 in mult_test_cases]
    ax_mult.set_xticks(mult_results['x_idx'])
    ax_mult.set_xticklabels(case_labels, rotation=45, ha='right', fontsize=8)
    
    ax_mult.set_xlabel(r'Test Case $(x_0, x_1)$')
    ax_mult.set_ylabel(r'Expected Value $\langle Z \rangle$')
    ax_mult.set_title(r'(b) Multiplication: $\langle Z \rangle = x_0 \cdot x_1$')
    ax_mult.legend(loc='best')
    ax_mult.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to: {save_path}")
    
    plt.show()
    
    return fig


def _format_polynomial(coefficients):
    """Format polynomial coefficients as LaTeX string."""
    terms = []
    for i, c in enumerate(coefficients):
        if abs(c) < 1e-10:
            continue
        if i == 0:
            terms.append(f"{c:.2f}")
        elif i == 1:
            sign = '+' if c >= 0 else ''
            terms.append(f"{sign}{c:.2f}x")
        else:
            sign = '+' if c >= 0 else ''
            terms.append(f"{sign}{c:.2f}x^{i}")
    return ''.join(terms) if terms else "0"


# ==============================================================================
# MAIN EXECUTION
# ==============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description='Generate polynomial recovery metrics plots with error bars'
    )
    parser.add_argument('--output-dir', type=str, default='.',
                       help='Output directory for figures')
    parser.add_argument('--format', type=str, default='pdf', 
                       choices=['pdf', 'png', 'svg'],
                       help='Output format')
    parser.add_argument('--n-points', type=int, default=15,
                       help='Number of test points')
    parser.add_argument('--n-trials', type=int, default=10,
                       help='Number of trials per point')
    parser.add_argument('--shots', type=int, default=2048,
                       help='Shots per circuit')
    parser.add_argument('--polynomial', type=float, nargs='+',
                       default=[0.1, 0.3, -0.1, 0.2],
                       help='Polynomial coefficients [a0, a1, a2, ...]')
    parser.add_argument('--operation-comparison', action='store_true',
                       help='Also generate operation comparison plot')
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    coefficients = np.array(args.polynomial)
    
    print("=" * 70)
    print("POLYNOMIAL RECOVERY METRICS")
    print("Neural-Native Quantum Arithmetic")
    print("=" * 70)
    print(f"Polynomial: F(x) = {_format_polynomial(coefficients)}")
    print(f"Test points: {args.n_points}")
    print(f"Trials per point: {args.n_trials}")
    print(f"Shots per circuit: {args.shots}")
    print("=" * 70)
    
    # Generate main recovery plot
    print("\nGenerating polynomial recovery plot...")
    fig1, results = create_recovery_plot(
        coefficients=coefficients,
        n_points=args.n_points,
        n_trials=args.n_trials,
        shots=args.shots,
        save_path=os.path.join(args.output_dir, f'polynomial_recovery.{args.format}')
    )
    
    # Generate operation comparison plot
    if args.operation_comparison:
        print("\nGenerating operation comparison plot...")
        fig2 = create_operation_comparison_plot(
            n_trials=args.n_trials,
            shots=args.shots,
            save_path=os.path.join(args.output_dir, f'operation_comparison.{args.format}')
        )
    
    # Print summary table
    print("\n" + "=" * 70)
    print("RECOVERY SUMMARY TABLE")
    print("=" * 70)
    print(f"{'x':>8} | {'tEV':>10} | {'mEV':>10} | {'std':>8} | {'error':>10} | {'Status':>8}")
    print("-" * 70)
    
    for i in range(len(results['x'])):
        x = results['x'][i]
        tEV = results['tEV'][i]
        mEV = results['mean_mEV'][i]
        std = results['std_mEV'][i]
        error = results['mean_error'][i]
        
        if abs(error) < 0.03:
            status = 'PASS'
        elif abs(error) < 0.1:
            status = 'POOR'
        else:
            status = 'FAIL'
        
        print(f"{x:>8.2f} | {tEV:>10.4f} | {mEV:>10.4f} | {std:>8.4f} | {error:>+10.4f} | {status:>8}")
    
    print("-" * 70)
    mean_abs_error = np.mean(np.abs(results['mean_error']))
    pass_rate = np.mean(np.abs(results['mean_error']) < 0.03) * 100
    print(f"Mean |error|: {mean_abs_error:.4f}")
    print(f"PASS rate: {pass_rate:.1f}%")
    print("=" * 70)


