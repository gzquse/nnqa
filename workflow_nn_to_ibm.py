#!/usr/bin/env python3
"""
Complete Workflow: Neural Network to IBM Quantum Cloud
=======================================================

End-to-end pipeline:
1. Train neural network for polynomial approximation
2. Extract learned coefficients
3. Build quantum circuits using quantum arithmetic
4. Submit to IBM Quantum Cloud (Heron processors)
5. Retrieve and analyze results
6. Generate publication-quality comparison plots

Usage:
    # Full workflow with IBM submission
    python workflow_nn_to_ibm.py --submit --backend ibm_boston
    
    # Local simulation only
    python workflow_nn_to_ibm.py --local
    
    # Retrieve and plot existing job
    python workflow_nn_to_ibm.py --retrieve --expName my_experiment
"""

import sys
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from pprint import pprint
from time import time
import json
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv
load_dotenv()

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit_aer import AerSimulator

# Import cloud submission utilities
from cloud_job.submit_nnqa_ibmq import (
    get_service, data_to_angle, build_polynomial_circuit,
    harvest_sampler_results, harvest_submit_meta
)
from toolbox.Util_ibm import harvest_circ_transpMeta
from toolbox.Util_H5io4 import write4_data_hdf5, read4_data_hdf5
from toolbox.Util_IOfunc import dateT2Str


# ==============================================================================
# NEURAL NETWORK MODEL
# ==============================================================================

class PolynomialNN(nn.Module):
    """Neural network that learns polynomial coefficients."""
    
    def __init__(self, degree=3):
        super().__init__()
        self.degree = degree
        self.coefficients = nn.Parameter(torch.zeros(degree + 1))
        
    def forward(self, x):
        if x.dim() == 2:
            x = x.squeeze(-1)
        result = torch.zeros_like(x)
        x_power = torch.ones_like(x)
        for i in range(self.degree + 1):
            result = result + self.coefficients[i] * x_power
            x_power = x_power * x
        return result
    
    def get_coefficients(self):
        return self.coefficients.detach().cpu().numpy()


class DeepPolynomialNN(nn.Module):
    """Deep neural network for polynomial approximation."""
    
    def __init__(self, hidden_dims=[32, 32], degree=3):
        super().__init__()
        self.degree = degree
        
        layers = []
        in_dim = 1
        for h_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(nn.Tanh())
            in_dim = h_dim
        layers.append(nn.Linear(in_dim, 1))
        
        self.network = nn.Sequential(*layers)
        
        # Polynomial fit layer for coefficient extraction
        self.poly_coeffs = None
    
    def forward(self, x):
        if x.dim() == 1:
            x = x.unsqueeze(-1)
        return self.network(x).squeeze(-1)
    
    def fit_polynomial(self, x_range=(-0.95, 0.95), n_points=100):
        """Fit polynomial to learned function."""
        x = torch.linspace(x_range[0], x_range[1], n_points)
        with torch.no_grad():
            y = self.forward(x).numpy()
        coeffs = np.polyfit(x.numpy(), y, self.degree)[::-1]
        self.poly_coeffs = coeffs
        return coeffs
    
    def get_coefficients(self):
        if self.poly_coeffs is None:
            self.fit_polynomial()
        return self.poly_coeffs


# ==============================================================================
# TRAINING
# ==============================================================================

def train_nn(model, target_func, epochs=300, lr=0.1, n_samples=200, verbose=True):
    """Train neural network on target function."""
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    
    X_train = torch.linspace(-0.95, 0.95, n_samples)
    y_train = torch.tensor([target_func(x.item()) for x in X_train], dtype=torch.float32)
    
    history = {'loss': [], 'epoch': []}
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        pred = model(X_train)
        loss = loss_fn(pred, y_train)
        loss.backward()
        optimizer.step()
        
        history['loss'].append(loss.item())
        history['epoch'].append(epoch)
        
        if verbose and epoch % 50 == 0:
            print(f"  Epoch {epoch:4d}: Loss = {loss.item():.6f}")
    
    return history


# ==============================================================================
# QUANTUM EXECUTION
# ==============================================================================

def run_quantum_local(x_values, coefficients, shots=8192):
    """Run quantum polynomial evaluation locally."""
    backend = AerSimulator()
    n_samples = len(x_values)
    
    measured = np.zeros(n_samples)
    measured_err = np.zeros(n_samples)
    theoretical = np.zeros(n_samples)
    
    for i, x in enumerate(x_values):
        # Build circuit
        qc, y_theo = build_polynomial_circuit(x, coefficients)
        theoretical[i] = y_theo
        
        # Execute
        qc_t = transpile(qc, backend, optimization_level=1)
        job = backend.run(qc_t, shots=shots)
        counts = job.result().get_counts()
        
        # Extract expectation value
        n0, n1 = counts.get('0', 0), counts.get('1', 0)
        mprob = n1 / (n0 + n1)
        measured[i] = 1 - 2 * mprob
        measured_err[i] = 2 * np.sqrt(mprob * (1 - mprob) / shots)
    
    return theoretical, measured, measured_err


def submit_to_ibm(x_values, coefficients, backend_name, shots, exp_name):
    """Submit quantum job to IBM cloud."""
    from qiskit_ibm_runtime import SamplerV2 as Sampler
    from qiskit_ibm_runtime.options.sampler_options import SamplerOptions
    from time import localtime
    
    service = get_service()
    backend = service.backend(backend_name)
    
    n_samples = len(x_values)
    theoretical = np.zeros(n_samples)
    
    # Build circuits
    circuits = []
    for i, x in enumerate(x_values):
        qc, y_theo = build_polynomial_circuit(x, coefficients)
        qc.metadata = {'sample_idx': i, 'x': float(x)}
        circuits.append(qc)
        theoretical[i] = y_theo
    
    # Transpile
    circuits_t = transpile(circuits, backend, optimization_level=3, seed_transpiler=42)
    
    # Setup sampler
    options = SamplerOptions()
    options.default_shots = shots
    options.twirling.enable_gates = True
    options.twirling.enable_measure = True
    options.twirling.num_randomizations = 60
    
    # Submit
    sampler = Sampler(mode=backend, options=options)
    job = sampler.run(tuple(circuits_t))
    
    # Build metadata
    expMD = {
        'payload': {
            'polynomial': coefficients.tolist(),
            'degree': len(coefficients) - 1,
            'num_sample': n_samples,
            'test_type': 'nn_polynomial',
        },
        'submit': {
            'num_shots': shots,
            'random_compilation': True,
            'dynamical_decoupling': False,
            'job_id': job.job_id(),
            'backend': backend_name,
            'date': dateT2Str(localtime()),
            'unix_time': int(time()),
            'provider': 'IBMQ_cloud',
        },
        'transpile': {},
        'postproc': {},
        'short_name': exp_name,
        'hash': job.job_id().replace('-', '')[3:9],
    }
    
    harvest_circ_transpMeta(circuits_t[0], expMD, backend_name)
    
    expD = {
        'x_values': x_values,
        'theoretical': theoretical,
        'coefficients': coefficients,
    }
    
    return job, expMD, expD


# ==============================================================================
# PUBLICATION-QUALITY PLOTTING
# ==============================================================================

def configure_publication_style():
    """Configure matplotlib for publication-quality figures."""
    import matplotlib.pyplot as plt
    
    plt.rcParams.update({
        # Font settings
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'DejaVu Serif', 'Computer Modern Roman'],
        'font.size': 11,
        'axes.labelsize': 13,
        'axes.titlesize': 14,
        'legend.fontsize': 10,
        'xtick.labelsize': 11,
        'ytick.labelsize': 11,
        
        # Figure size for single column (3.5") or double column (7")
        'figure.figsize': (7, 5),
        'figure.dpi': 150,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.05,
        
        # Axes
        'axes.linewidth': 1.0,
        'axes.grid': False,
        'axes.spines.top': False,
        'axes.spines.right': False,
        
        # Lines
        'lines.linewidth': 2.0,
        'lines.markersize': 7,
        
        # Legend
        'legend.framealpha': 0.95,
        'legend.edgecolor': '0.8',
        'legend.fancybox': False,
        'legend.frameon': True,
        
        # Grid
        'grid.alpha': 0.3,
        'grid.linewidth': 0.5,
    })


def create_comparison_plot(x_values, theoretical, classical_pred, quantum_meas, 
                          quantum_err, coefficients, metadata, save_path=None):
    """
    Create publication-quality comparison plot:
    - Classical NN prediction
    - Quantum measurement with error bars
    - Theoretical polynomial
    """
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    
    configure_publication_style()
    
    # IBM Design colorblind-friendly palette
    colors = {
        'theoretical': '#2E8B57',     # Sea Green
        'classical': '#648FFF',       # Blue
        'quantum': '#DC267F',         # Magenta
        'error_fill': '#FFB000',      # Gold
        'residual': '#785EF0',        # Purple
    }
    
    # Create figure with GridSpec
    fig = plt.figure(figsize=(10, 8))
    gs = gridspec.GridSpec(2, 2, height_ratios=[2, 1], hspace=0.3, wspace=0.3)
    
    # Dense x for smooth curves
    x_dense = np.linspace(-1, 1, 200)
    y_dense = np.array([sum(coefficients[i] * (x ** i) for i in range(len(coefficients))) 
                        for x in x_dense])
    
    # ==========================================================================
    # Panel A: Polynomial Recovery Comparison
    # ==========================================================================
    ax1 = fig.add_subplot(gs[0, 0])
    
    # Theoretical curve
    ax1.plot(x_dense, y_dense, color=colors['theoretical'], linewidth=2.5,
            label='Target Polynomial', zorder=2)
    
    # Classical NN prediction
    ax1.scatter(x_values, classical_pred, color=colors['classical'], 
               s=60, marker='s', edgecolors='black', linewidth=0.5,
               label='Classical NN', zorder=3, alpha=0.8)
    
    # Quantum measurement with error bars
    ax1.errorbar(x_values, quantum_meas, yerr=quantum_err,
                fmt='o', color=colors['quantum'], markersize=8,
                capsize=4, capthick=1.5, elinewidth=1.5,
                markeredgecolor='black', markeredgewidth=0.5,
                label='Quantum Circuit', zorder=4)
    
    ax1.set_xlabel(r'Input $x$')
    ax1.set_ylabel(r'Output $F(x)$')
    ax1.set_title('(a) Polynomial Recovery Comparison')
    ax1.legend(loc='best', framealpha=0.9)
    ax1.set_xlim(-1.05, 1.05)
    ax1.grid(True, alpha=0.3)
    
    # Add polynomial formula
    poly_str = format_polynomial(coefficients)
    ax1.text(0.03, 0.97, f'$F(x) = {poly_str}$',
            transform=ax1.transAxes, fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='wheat', alpha=0.8))
    
    # ==========================================================================
    # Panel B: Correlation Plot
    # ==========================================================================
    ax2 = fig.add_subplot(gs[0, 1])
    
    # Classical correlation
    ax2.scatter(theoretical, classical_pred, color=colors['classical'],
               s=50, marker='s', alpha=0.7, label='Classical NN',
               edgecolors='black', linewidth=0.3)
    
    # Quantum correlation with error bars
    ax2.errorbar(theoretical, quantum_meas, yerr=quantum_err,
                fmt='o', color=colors['quantum'], markersize=7,
                capsize=3, capthick=1, elinewidth=1,
                markeredgecolor='black', markeredgewidth=0.3,
                alpha=0.9, label='Quantum Circuit')
    
    # Ideal line
    lim = max(abs(theoretical.min()), abs(theoretical.max())) + 0.1
    ax2.plot([-lim, lim], [-lim, lim], 'k--', linewidth=1.5, label='Ideal', alpha=0.7)
    
    # Compute correlations
    corr_classical = np.corrcoef(theoretical, classical_pred)[0, 1]
    corr_quantum = np.corrcoef(theoretical, quantum_meas)[0, 1]
    
    ax2.set_xlabel('Theoretical Value')
    ax2.set_ylabel('Predicted/Measured Value')
    ax2.set_title('(b) Correlation Analysis')
    ax2.set_xlim(-lim, lim)
    ax2.set_ylim(-lim, lim)
    ax2.set_aspect('equal')
    ax2.legend(loc='lower right', fontsize=9)
    ax2.grid(True, alpha=0.3)
    
    # Correlation text
    corr_text = f'Classical: $r$ = {corr_classical:.4f}\nQuantum: $r$ = {corr_quantum:.4f}'
    ax2.text(0.05, 0.95, corr_text, transform=ax2.transAxes,
            fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.9))
    
    # ==========================================================================
    # Panel C: Recovery Error Distribution
    # ==========================================================================
    ax3 = fig.add_subplot(gs[1, 0])
    
    classical_error = classical_pred - theoretical
    quantum_error = quantum_meas - theoretical
    
    # Histograms
    bins = np.linspace(-0.15, 0.15, 25)
    ax3.hist(classical_error, bins=bins, color=colors['classical'], alpha=0.6,
            label=f'Classical (RMSE={np.sqrt(np.mean(classical_error**2)):.4f})',
            edgecolor='black', linewidth=0.5)
    ax3.hist(quantum_error, bins=bins, color=colors['quantum'], alpha=0.6,
            label=f'Quantum (RMSE={np.sqrt(np.mean(quantum_error**2)):.4f})',
            edgecolor='black', linewidth=0.5)
    
    ax3.axvline(0, color='black', linestyle='-', linewidth=1.5)
    
    # Threshold regions
    ax3.axvspan(-0.03, 0.03, color='green', alpha=0.15, label='PASS region')
    
    ax3.set_xlabel('Recovery Error (Predicted - Theoretical)')
    ax3.set_ylabel('Count')
    ax3.set_title('(c) Error Distribution')
    ax3.legend(loc='upper right', fontsize=8)
    ax3.set_xlim(-0.15, 0.15)
    
    # ==========================================================================
    # Panel D: Error vs Input
    # ==========================================================================
    ax4 = fig.add_subplot(gs[1, 1])
    
    ax4.scatter(x_values, classical_error, color=colors['classical'],
               s=50, marker='s', alpha=0.7, label='Classical NN',
               edgecolors='black', linewidth=0.3)
    
    ax4.errorbar(x_values, quantum_error, yerr=quantum_err,
                fmt='o', color=colors['quantum'], markersize=7,
                capsize=3, capthick=1, elinewidth=1,
                markeredgecolor='black', markeredgewidth=0.3,
                alpha=0.9, label='Quantum Circuit')
    
    ax4.axhline(0, color='black', linestyle='-', linewidth=1)
    ax4.axhspan(-0.03, 0.03, color='green', alpha=0.15)
    
    ax4.set_xlabel(r'Input $x$')
    ax4.set_ylabel('Recovery Error')
    ax4.set_title('(d) Error vs Input')
    ax4.legend(loc='upper right', fontsize=9)
    ax4.set_xlim(-1.05, 1.05)
    ax4.set_ylim(-0.15, 0.15)
    ax4.grid(True, alpha=0.3)
    
    # ==========================================================================
    # Add metadata annotation
    # ==========================================================================
    meta_text = (f"Backend: {metadata.get('backend', 'Local')}\n"
                f"Shots: {metadata.get('shots', 'N/A')}\n"
                f"Samples: {len(x_values)}")
    fig.text(0.99, 0.01, meta_text, fontsize=8, ha='right', va='bottom',
            transform=fig.transFigure,
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to: {save_path}")
        # Also save PDF version
        pdf_path = save_path.rsplit('.', 1)[0] + '.pdf'
        plt.savefig(pdf_path, dpi=300, bbox_inches='tight')
        print(f"PDF saved to: {pdf_path}")
    
    return fig


def create_workflow_summary_plot(workflow_results, save_path=None):
    """Create summary plot showing the full NN-to-Quantum workflow."""
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    
    configure_publication_style()
    
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    
    colors = {
        'nn': '#648FFF',
        'quantum': '#DC267F',
        'target': '#2E8B57',
    }
    
    # Panel 1: Training Loss
    ax1 = axes[0]
    history = workflow_results['training_history']
    ax1.semilogy(history['epoch'], history['loss'], color=colors['nn'], linewidth=2)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('MSE Loss')
    ax1.set_title('(a) Neural Network Training')
    ax1.grid(True, alpha=0.3)
    
    final_loss = history['loss'][-1]
    ax1.text(0.95, 0.95, f'Final Loss: {final_loss:.2e}',
            transform=ax1.transAxes, ha='right', va='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
    
    # Panel 2: Coefficient Comparison
    ax2 = axes[1]
    true_coeffs = workflow_results['true_coefficients']
    learned_coeffs = workflow_results['learned_coefficients']
    
    x_pos = np.arange(len(true_coeffs))
    width = 0.35
    
    bars1 = ax2.bar(x_pos - width/2, true_coeffs, width, color=colors['target'],
                   label='Target', edgecolor='black', linewidth=0.5)
    bars2 = ax2.bar(x_pos + width/2, learned_coeffs, width, color=colors['nn'],
                   label='Learned', edgecolor='black', linewidth=0.5)
    
    ax2.set_xlabel('Coefficient Index')
    ax2.set_ylabel('Value')
    ax2.set_title('(b) Coefficient Recovery')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels([f'$a_{i}$' for i in range(len(true_coeffs))])
    ax2.legend(loc='best')
    ax2.axhline(0, color='black', linewidth=0.5)
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Panel 3: Quantum vs Classical RMSE
    ax3 = axes[2]
    
    methods = ['Classical NN', 'Quantum Circuit']
    rmse_values = [
        workflow_results['classical_rmse'],
        workflow_results['quantum_rmse']
    ]
    bar_colors = [colors['nn'], colors['quantum']]
    
    bars = ax3.bar(methods, rmse_values, color=bar_colors, edgecolor='black', linewidth=1)
    
    # Add value labels
    for bar, val in zip(bars, rmse_values):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
                f'{val:.4f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    ax3.set_ylabel('RMSE')
    ax3.set_title('(c) Recovery Accuracy')
    ax3.set_ylim(0, max(rmse_values) * 1.3)
    ax3.grid(True, alpha=0.3, axis='y')
    
    # Threshold line
    ax3.axhline(0.03, color='green', linestyle='--', linewidth=1.5, label='PASS threshold')
    ax3.legend(loc='upper right')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Workflow summary saved to: {save_path}")
    
    return fig


def format_polynomial(coefficients):
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
# MAIN WORKFLOW
# ==============================================================================

def run_workflow(args):
    """Execute the complete NN to Quantum workflow."""
    
    print("=" * 70)
    print("NEURAL NETWORK TO IBM QUANTUM WORKFLOW")
    print("=" * 70)
    
    # Setup output directory
    output_dir = os.path.join(args.output_dir, f'workflow_{args.exp_name}')
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'plots'), exist_ok=True)
    
    # ==========================================================================
    # Step 1: Define Target Polynomial
    # ==========================================================================
    print("\n[Step 1] Define Target Polynomial")
    print("-" * 50)
    
    true_coeffs = np.array(args.polynomial)
    
    def target_func(x):
        return sum(true_coeffs[i] * (x ** i) for i in range(len(true_coeffs)))
    
    print(f"Target: F(x) = {format_polynomial(true_coeffs)}")
    
    # Verify output range is in [-1, 1]
    x_check = np.linspace(-1, 1, 100)
    y_check = np.array([target_func(x) for x in x_check])
    print(f"Output range: [{y_check.min():.3f}, {y_check.max():.3f}]")
    
    if y_check.max() > 1 or y_check.min() < -1:
        print("WARNING: Output exceeds [-1, 1] range. Clipping will occur.")
    
    # ==========================================================================
    # Step 2: Train Neural Network
    # ==========================================================================
    print("\n[Step 2] Train Neural Network")
    print("-" * 50)
    
    degree = len(true_coeffs) - 1
    model = PolynomialNN(degree=degree)
    
    history = train_nn(model, target_func, epochs=args.epochs, 
                      lr=args.lr, n_samples=args.train_samples)
    
    learned_coeffs = model.get_coefficients()
    print(f"\nLearned coefficients: {learned_coeffs}")
    print(f"True coefficients:    {true_coeffs}")
    print(f"Coefficient error:    {np.abs(learned_coeffs - true_coeffs)}")
    
    # ==========================================================================
    # Step 3: Generate Test Points
    # ==========================================================================
    print("\n[Step 3] Generate Test Points")
    print("-" * 50)
    
    np.random.seed(args.seed)
    x_values = np.linspace(-0.9, 0.9, args.num_samples)
    
    # Classical predictions
    classical_pred = np.array([
        sum(learned_coeffs[i] * (x ** i) for i in range(len(learned_coeffs)))
        for x in x_values
    ])
    
    # Theoretical values
    theoretical = np.array([
        np.clip(target_func(x), -1 + 1e-6, 1 - 1e-6)
        for x in x_values
    ])
    
    print(f"Test points: {len(x_values)}")
    
    # ==========================================================================
    # Step 4: Quantum Execution
    # ==========================================================================
    print("\n[Step 4] Quantum Execution")
    print("-" * 50)
    
    if args.local:
        print("Running local quantum simulation...")
        theo_q, quantum_meas, quantum_err = run_quantum_local(
            x_values, learned_coeffs, shots=args.shots
        )
        metadata = {'backend': 'AerSimulator', 'shots': args.shots}
        
    elif args.submit:
        print(f"Submitting to IBM Quantum: {args.backend}")
        job, expMD, expD = submit_to_ibm(
            x_values, learned_coeffs, args.backend, args.shots, args.exp_name
        )
        
        # Save job info
        job_file = os.path.join(output_dir, f'{args.exp_name}.ibm.h5')
        expD['classical_pred'] = classical_pred
        expD['learned_coefficients'] = learned_coeffs
        expD['true_coefficients'] = true_coeffs
        write4_data_hdf5(expD, job_file, expMD)
        
        print(f"\nJob submitted: {job.job_id()}")
        print(f"Backend: {args.backend}")
        print(f"Saved to: {job_file}")
        print(f"\nTo retrieve results, run:")
        print(f"  python workflow_nn_to_ibm.py --retrieve --exp-name {args.exp_name}")
        
        # Wait for results if requested
        if args.wait:
            print("\nWaiting for job completion...")
            from time import sleep
            while True:
                status = job.status()
                print(f"  Status: {status}")
                if status == 'DONE':
                    break
                if status == 'ERROR':
                    print("Job failed!")
                    return
                sleep(30)
            
            # Harvest results
            harvest_sampler_results(job, expMD, expD)
            quantum_meas = expD['measured']
            quantum_err = expD['measured_err']
            metadata = {'backend': args.backend, 'shots': args.shots}
        else:
            return  # Exit and wait for manual retrieval
            
    elif args.retrieve:
        print(f"Retrieving results for: {args.exp_name}")
        job_file = os.path.join(output_dir, f'{args.exp_name}.ibm.h5')
        
        if not os.path.exists(job_file):
            # Try cloud_job/out/jobs
            job_file = os.path.join('cloud_job/out/jobs', f'{args.exp_name}.ibm.h5')
        
        expD, expMD = read4_data_hdf5(job_file)
        
        service = get_service()
        job = service.job(expMD['submit']['job_id'])
        
        print(f"Job status: {job.status()}")
        if job.status() != 'DONE':
            print("Job not complete yet.")
            return
        
        harvest_sampler_results(job, expMD, expD)
        
        quantum_meas = expD['measured']
        quantum_err = expD['measured_err']
        theoretical = expD['theoretical']
        x_values = expD['x_values']
        classical_pred = expD.get('classical_pred', theoretical)
        learned_coeffs = expD.get('learned_coefficients', true_coeffs)
        
        metadata = {'backend': expMD['submit']['backend'], 'shots': expMD['submit']['num_shots']}
        
        # Save measurement file
        meas_file = os.path.join(output_dir, f'{args.exp_name}.meas.h5')
        write4_data_hdf5(expD, meas_file, expMD)
    
    # ==========================================================================
    # Step 5: Analysis and Plotting
    # ==========================================================================
    print("\n[Step 5] Analysis and Plotting")
    print("-" * 50)
    
    # Compute metrics
    classical_rmse = np.sqrt(np.mean((classical_pred - theoretical) ** 2))
    quantum_rmse = np.sqrt(np.mean((quantum_meas - theoretical) ** 2))
    
    classical_corr = np.corrcoef(theoretical, classical_pred)[0, 1]
    quantum_corr = np.corrcoef(theoretical, quantum_meas)[0, 1]
    
    print(f"Classical NN RMSE:     {classical_rmse:.6f}")
    print(f"Quantum Circuit RMSE:  {quantum_rmse:.6f}")
    print(f"Classical Correlation: {classical_corr:.6f}")
    print(f"Quantum Correlation:   {quantum_corr:.6f}")
    
    # Workflow results for summary plot
    workflow_results = {
        'training_history': history,
        'true_coefficients': true_coeffs,
        'learned_coefficients': learned_coeffs,
        'classical_rmse': classical_rmse,
        'quantum_rmse': quantum_rmse,
    }
    
    # Create comparison plot
    comparison_path = os.path.join(output_dir, 'plots', 
                                   f'{args.exp_name}_comparison.png')
    fig1 = create_comparison_plot(
        x_values, theoretical, classical_pred, quantum_meas, quantum_err,
        true_coeffs, metadata, save_path=comparison_path
    )
    
    # Create workflow summary plot
    summary_path = os.path.join(output_dir, 'plots',
                                f'{args.exp_name}_workflow_summary.png')
    fig2 = create_workflow_summary_plot(workflow_results, save_path=summary_path)
    
    # ==========================================================================
    # Save Results
    # ==========================================================================
    print("\n[Step 6] Save Results")
    print("-" * 50)
    
    results = {
        'x_values': x_values.tolist(),
        'theoretical': theoretical.tolist(),
        'classical_pred': classical_pred.tolist(),
        'quantum_meas': quantum_meas.tolist(),
        'quantum_err': quantum_err.tolist(),
        'true_coefficients': true_coeffs.tolist(),
        'learned_coefficients': learned_coeffs.tolist(),
        'metrics': {
            'classical_rmse': classical_rmse,
            'quantum_rmse': quantum_rmse,
            'classical_correlation': classical_corr,
            'quantum_correlation': quantum_corr,
        },
        'metadata': metadata,
    }
    
    results_file = os.path.join(output_dir, f'{args.exp_name}_results.json')
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Results saved to: {results_file}")
    print(f"Plots saved to: {os.path.join(output_dir, 'plots')}")
    
    print("\n" + "=" * 70)
    print("WORKFLOW COMPLETE")
    print("=" * 70)
    
    import matplotlib.pyplot as plt
    if not args.no_display:
        plt.show()


# ==============================================================================
# ARGUMENT PARSER
# ==============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description='Complete NN to IBM Quantum Workflow',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Local simulation
  python workflow_nn_to_ibm.py --local --exp-name demo_local
  
  # Submit to IBM Boston
  python workflow_nn_to_ibm.py --submit --backend ibm_boston --exp-name demo_boston
  
  # Submit and wait for results
  python workflow_nn_to_ibm.py --submit --backend ibm_fez --wait --exp-name demo_fez
  
  # Retrieve existing job
  python workflow_nn_to_ibm.py --retrieve --exp-name demo_boston
        """
    )
    
    # Execution mode
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--local', action='store_true', help='Run local simulation')
    mode.add_argument('--submit', action='store_true', help='Submit to IBM Quantum')
    mode.add_argument('--retrieve', action='store_true', help='Retrieve existing job')
    
    # Polynomial
    parser.add_argument('-p', '--polynomial', nargs='+', type=float,
                       default=[0.1, 0.3, -0.1, 0.2],
                       help='Polynomial coefficients [a0, a1, a2, ...]')
    
    # Training
    parser.add_argument('--epochs', type=int, default=300, help='Training epochs')
    parser.add_argument('--lr', type=float, default=0.1, help='Learning rate')
    parser.add_argument('--train-samples', type=int, default=200, help='Training samples')
    
    # Quantum
    parser.add_argument('--num-samples', type=int, default=15, help='Test samples')
    parser.add_argument('--shots', type=int, default=4096, help='Shots per circuit')
    parser.add_argument('-b', '--backend', default='ibm_boston',
                       help='IBM backend')
    parser.add_argument('--wait', action='store_true', help='Wait for job completion')
    
    # Output
    parser.add_argument('--exp-name', default='nn_quantum_demo', help='Experiment name')
    parser.add_argument('--output-dir', default='results', help='Output directory')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--no-display', action='store_true', help='Do not show plots')
    
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    run_workflow(args)

